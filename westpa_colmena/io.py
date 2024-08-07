"""HDF5 I/O for WESTPA simulations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import westpa_colmena
from westpa_colmena.ensemble import BasisStates
from westpa_colmena.ensemble import IterationMetadata
from westpa_colmena.ensemble import SimMetadata
from westpa_colmena.ensemble import TargetState

# Define data types for use in the HDF5 file

# Up to 9 quintillion segments per iteration;
# signed so that initial states can be stored negative
seg_id_dtype = np.int64
# Up to 4 billion iterations
n_iter_dtype = np.uint32
# About 15 digits of precision in weights
weight_dtype = np.float64
# ("u" for Unix time) Up to ~10^300 cpu-seconds
utime_dtype = np.float64
# Variable-length string
vstr_dtype = h5py.special_dtype(vlen=str)
# Reference to an HDF5 object
h5ref_dtype = h5py.special_dtype(ref=h5py.Reference)
# Hash of a binning scheme
binhash_dtype = np.dtype('|S64')

summary_table_dtype = np.dtype(
    [
        # Number of live trajectories in this iteration
        ('n_particles', seg_id_dtype),
        # Norm of probability, to watch for errors or drift
        ('norm', weight_dtype),
        # Per-bin minimum probability
        ('min_bin_prob', weight_dtype),
        # Per-bin maximum probability
        ('max_bin_prob', weight_dtype),
        # Per-segment minimum probability
        ('min_seg_prob', weight_dtype),
        # Per-segment maximum probability
        ('max_seg_prob', weight_dtype),
        # Total CPU time for this iteration
        ('cputime', utime_dtype),
        # Total wallclock time for this iteration
        ('walltime', utime_dtype),
        # Hash of the binning scheme used in this iteration
        ('binhash', binhash_dtype),
    ],
)

# Index to basis/initial states
ibstate_index_dtype = np.dtype(
    [
        # Iteration when this state list is valid
        ('iter_valid', np.uint),
        # Number of basis states
        ('n_bstates', np.uint),
        # Reference to a group containing further data
        ('group_ref', h5ref_dtype),
    ],
)

# Basis state index type
bstate_dtype = np.dtype(
    [
        # An optional descriptive label
        ('label', vstr_dtype),
        # Probability that this state will be selected
        ('probability', weight_dtype),
        # An optional auxiliary data reference
        ('auxref', vstr_dtype),
    ],
)

tstate_index_dtype = np.dtype(
    [
        # Iteration when this state list is valid
        ('iter_valid', np.uint),
        # Number of target states
        ('n_states', np.uint),
        # Reference to a group containing further data; this will be the
        ('group_ref', h5ref_dtype),
    ],
)

# Null reference if there is no target state for that timeframe.
# An optional descriptive label for this state
tstate_dtype = np.dtype([('label', vstr_dtype)])

# Storage of bin identities
binning_index_dtype = np.dtype(
    [('hash', binhash_dtype), ('pickle_len', np.uint32)],
)

seg_status_dtype = np.uint8
seg_endpoint_dtype = np.uint8

# The HDF5 file tracks two distinct, but related, histories:
#    (1) the evolution of the trajectory, which requires only an identifier
#        of where a segment's initial state comes from (the "history graph");
#        this is stored as the parent_id field of the seg index
#    (2) the flow of probability due to splits, merges, and recycling events,
#        which can be thought of as an adjacency list (the "weight graph")
# segment ID is implied by the row in the index table, and so is not stored
# initpoint_type remains implicitly stored as negative IDs (if parent_id < 0,
# then init_state_id = -(parent_id+1)
seg_index_dtype = np.dtype(
    [
        # Statistical weight of this segment
        ('weight', weight_dtype),
        # ID of parent (for trajectory history)
        ('parent_id', seg_id_dtype),
        # number of parents this segment has in the weight transfer graph
        ('wtg_n_parents', np.uint),
        # offset into the weight transfer graph dataset
        ('wtg_offset', np.uint),
        # CPU time used in propagating this segment
        ('cputime', utime_dtype),
        # Wallclock time used in propagating this segment
        ('walltime', utime_dtype),
        # Endpoint type (will continue, merged, or recycled)
        ('endpoint_type', seg_endpoint_dtype),
        # Status of propagation of this segment
        ('status', seg_status_dtype),
    ],
)


class WestpaH5File:
    """Utility class for writing WESTPA HDF5 files."""

    # Default metadata for the WESTPA HDF5 file
    west_fileformat_version: int = 9
    west_iter_prec: int = 8
    west_version: str = westpa_colmena.__version__

    def __init__(self, westpa_h5file_path: str | Path) -> None:
        self.westpa_h5file_path = westpa_h5file_path

        # Initialize the HDF5 file if it does not exist
        if not Path(self.westpa_h5file_path).exists():
            self._initialize_hdf5_file()

    def _initialize_hdf5_file(self) -> None:
        """Initialize the HDF5 file."""
        # Create the file
        with h5py.File(self.westpa_h5file_path, mode='w') as f:
            # Set attribute metadata
            f.attrs['west_file_format_version'] = self.west_fileformat_version
            f.attrs['west_iter_prec'] = self.west_iter_prec
            f.attrs['west_version'] = self.west_version
            f.attrs['westpa_iter_prec'] = self.west_iter_prec
            f.attrs['westpa_fileformat_version'] = self.west_fileformat_version
            f.attrs['west_current_iteration'] = 1  # WESTPA is 1-indexed

            # Create the summary table
            f.create_dataset(
                'summary',
                shape=(0,),
                dtype=summary_table_dtype,
                maxshape=(None,),
            )

            # Create the iterations group
            f.create_group('iterations')

    def _find_multi_iter_group(
        self,
        h5_file: h5py.File,
        n_iter: int,
        group_name: str,
    ) -> h5py.Group | None:
        """Find the group for the specified iteration.

        This function is borrowed from the WESTPA codebase and is used to
        read the group for the specified iteration out of the HDF5 file.
        """
        group = h5_file[group_name]
        index = group['index'][...]
        set_id = np.digitize([n_iter], index['iter_valid']) - 1
        group_ref = index[set_id]['group_ref']

        # Check if reference is Null
        if not bool(group_ref):
            return None

        # This extra [0] is to work around a bug in h5py
        try:
            group = h5_file[group_ref]
        except (TypeError, AttributeError):
            group = h5_file[group_ref[0]]
        return group

    def _append_summary(
        self,
        h5_file: h5py.File,
        n_iter: int,
        cur_sims: list[SimMetadata],
        metadata: IterationMetadata,
    ) -> None:
        """Create a row for the summary table."""
        # Create a row for the summary table
        summary_row = np.zeros((1,), dtype=summary_table_dtype)
        # The number of simulation segments in this iteration
        summary_row['n_particles'] = len(cur_sims)
        # Compute the total weight of all segments (should be close to 1.0)
        summary_row['norm'] = sum(x.weight for x in cur_sims)
        # Compute the min and max weight of each bin
        summary_row['min_bin_prob'] = metadata.min_bin_prob
        summary_row['max_bin_prob'] = metadata.max_bin_prob
        # Compute the min and max weight over all segments
        summary_row['min_seg_prob'] = min(x.weight for x in cur_sims)
        summary_row['max_seg_prob'] = max(x.weight for x in cur_sims)

        # Compute the total CPU time for this iteration (in seconds)
        summary_row['cputime'] = sum(x.cputime for x in cur_sims)
        # TODO: This should be the total workflow overhead time
        # for running an iteration. This is not currently tracked.
        # Update this once we refactor the thinker.

        # Compute the total wallclock time for this iteration (in seconds)
        summary_row['walltime'] = max(x.walltime for x in cur_sims)

        # Save a hex string identifying the binning used in this iteration
        summary_row['binhash'] = metadata.binner_hash

        # Create a table of summary information about each iteration
        summary_table = h5_file['summary']

        # Update the summary table
        summary_table.resize((len(summary_table) + 1,))
        summary_table[n_iter - 1] = summary_row

    def _append_ibstates(
        self,
        h5_file: h5py.File,
        n_iter: int,
        basis_states: BasisStates,
    ) -> None:
        """Append the initial basis states to the HDF5 file."""
        # Create the group used to store basis states and initial states
        group = h5_file.require_group('ibstates')

        # Check if 'index' dataset exists in group
        if 'index' in group:
            # Resize the index dataset to add a new row
            index = group['index']
            index.resize((len(index) + 1,))
        else:
            # Create the index dataset if it does not exist
            index = group.create_dataset(
                'index',
                dtype=ibstate_index_dtype,
                shape=(1,),
                maxshape=(None,),
            )

        # Get the unique basis states
        unique_bstates = basis_states.unique_basis_states

        # Create a new row for the index dataset
        set_id = len(index) - 1
        index_row = index[set_id]
        index_row['iter_valid'] = n_iter - 1
        index_row['n_bstates'] = len(unique_bstates)
        state_group = group.create_group(str(set_id))
        index_row['group_ref'] = state_group.ref

        if unique_bstates:
            # Create the basis state table
            state_table = np.empty((len(unique_bstates),), dtype=bstate_dtype)

            # Populate the state table
            for i, state in enumerate(unique_bstates):
                state_table[i]['label'] = str(state.simulation_id)
                # NOTE: This assumes that all basis states are equally likely
                #       this is implicitly coupled to the
                #       BasisStates._uniform_init function.
                state_table[i]['probability'] = 1.0 / len(unique_bstates)
                state_table[i]['auxref'] = state.parent_restart_file.name

            # Get the pcoords for the basis states
            state_pcoords = [x.parent_pcoord for x in unique_bstates]
            state_pcoords = np.array(state_pcoords, dtype=np.float32)

            # Add the basis state table to the state group
            state_group['bstate_index'] = state_table
            state_group['bstate_pcoord'] = state_pcoords

        # Update the index dataset
        index[set_id] = index_row

    def _append_tstates(
        self,
        h5_file: h5py.File,
        n_iter: int,
        target_states: list[TargetState],
    ) -> None:
        """Append the target states to the HDF5 file."""
        # Create the group used to store target states
        group = h5_file.require_group('tstates')

        if 'index' in group:
            # Resize the index dataset to add a new row
            index = group['index']
            index.resize((len(index) + 1,))
        else:
            # Create the index dataset if it does not exist
            index = group.create_dataset(
                'index',
                dtype=tstate_index_dtype,
                shape=(1,),
                maxshape=(None,),
            )

        # Create a new row for the index dataset
        set_id = len(index) - 1
        index_row = index[set_id]
        index_row['iter_valid'] = n_iter - 1
        index_row['n_states'] = len(target_states)

        if target_states:
            # Collect the target state labels
            state_table = np.empty((len(target_states),), dtype=tstate_dtype)
            for i, state in enumerate(target_states):
                state_table[i]['label'] = state.label

            # Collect the pcoords for the target states
            state_pcoords = [x.pcoord for x in target_states]
            state_pcoords = np.array(state_pcoords, dtype=np.float32)

            # Create the group for the target states
            state_group = group.create_group(str(set_id))

            # Add the target state table to the state group
            index_row['group_ref'] = state_group.ref
            state_group['index'] = state_table
            state_group['pcoord'] = state_pcoords

        else:
            index_row['group_ref'] = None

        # Update the index dataset
        index[set_id] = index_row

    def _append_bin_mapper(
        self,
        h5_file: h5py.File,
        metadata: IterationMetadata,
    ) -> None:
        """Append the bin mapper to the HDF5 file."""
        # Create the group used to store bin mapper
        group = h5_file.require_group('bin_topologies')

        # Extract the bin mapper data
        pickle_data = metadata.binner_pickle
        hashval = metadata.binner_hash

        if 'index' in group and 'pickles' in group:
            # Resize the index and pickle_ds datasets to add a new row
            index = group['index']
            pickle_ds = group['pickles']
            index.resize((len(index) + 1,))
            new_hsize = max(pickle_ds.shape[1], len(pickle_data))
            pickle_ds.resize((len(pickle_ds) + 1, new_hsize))
        else:
            # Create the index and pickle_ds datasets if they do not exist
            index = group.create_dataset(
                'index',
                shape=(1,),
                maxshape=(None,),
                dtype=binning_index_dtype,
            )
            pickle_ds = group.create_dataset(
                'pickles',
                dtype=np.uint8,
                shape=(1, len(pickle_data)),
                maxshape=(None, None),
                chunks=(1, 4096),
                compression='gzip',
                compression_opts=9,
            )

        # Populate the new row in the index dataset
        ind = len(index) - 1
        index_row = index[ind]
        index_row['hash'] = hashval
        index_row['pickle_len'] = len(pickle_data)

        # Update the index and pickle_ds datasets
        index[ind] = index_row
        pickle_ds[ind, : len(pickle_data)] = memoryview(pickle_data)

    def _append_seg_index_table(
        self,
        iter_group: h5py.Group,
        cur_sims: list[SimMetadata],
    ) -> None:
        """Append the seg_index table to the HDF5 file."""
        # Create the seg index table
        seg_index = np.empty((len(cur_sims),), dtype=seg_index_dtype)

        # Populate the seg index table
        total_parents = 0
        for idx, sim in enumerate(cur_sims):
            seg_index[idx]['weight'] = sim.weight
            seg_index[idx]['parent_id'] = sim.parent_simulation_id
            seg_index[idx]['wtg_n_parents'] = len(sim.wtg_parent_ids)
            seg_index[idx]['wtg_offset'] = total_parents
            seg_index[idx]['cputime'] = sim.cputime
            seg_index[idx]['walltime'] = sim.walltime
            seg_index[idx]['endpoint_type'] = sim.endpoint_type
            # We set status to 2 to indicate the sim is complete
            seg_index[idx]['status'] = 2
            total_parents += len(sim.wtg_parent_ids)

        # Write the seg_index dataset
        iter_group.create_dataset('seg_index', data=seg_index)

        # Create the weight transfer graph dataset
        wtg_parent_ids = []
        for sim in cur_sims:
            wtg_parent_ids.extend(list(sim.wtg_parent_ids))
        wtg_parent_ids = np.array(wtg_parent_ids, dtype=seg_id_dtype)

        # Write the wtgraph dataset
        iter_group.create_dataset('wtgraph', data=wtg_parent_ids)

    def _append_pcoords(
        self,
        iter_group: h5py.Group,
        cur_sims: list[SimMetadata],
    ) -> None:
        """Append the pcoords to the HDF5 file."""
        # Extract the pcoords with shape (n_sims, 1 + n_frames, pcoord_ndim)
        # where the first frame is the parent pcoord and the rest are the
        # pcoords for each frame in the current simulation.
        pcoords = np.array(
            [[x.parent_pcoord, *x.pcoord] for x in cur_sims],
            dtype=np.float32,
        )

        # Create the pcoord dataset
        iter_group.create_dataset('pcoord', data=pcoords)

    def _append_bin_target_counts(
        self,
        iter_group: h5py.Group,
        metadata: IterationMetadata,
    ) -> None:
        """Append the bin_target_counts to the HDF5 file."""
        # Create the bin_target_counts dataset
        counts = np.array(metadata.bin_target_counts, dtype=np.float16)
        iter_group.create_dataset('bin_target_counts', data=counts)

    def _append_iter_ibstates(
        self,
        hf_file: h5py.File,
        iter_group: h5py.Group,
        n_iter: int,
    ) -> None:
        """Append the ibstates datasets for the current iteration."""
        # Create the ibstates datasets for the current iteration
        iter_group['ibstates'] = self._find_multi_iter_group(
            hf_file,
            n_iter,
            'ibstates',
        )

    def _append_iter_tstates(
        self,
        hf_file: h5py.File,
        iter_group: h5py.Group,
        n_iter: int,
    ) -> None:
        """Append the tstates datasets for the current iteration."""
        # Create the tstates datasets for the current iteration
        tstate_group = self._find_multi_iter_group(
            hf_file,
            n_iter,
            'tstates',
        )
        if tstate_group is not None:
            iter_group['tstates'] = tstate_group

    def _append_auxdata(
        self,
        iter_group: h5py.Group,
        cur_sims: list[SimMetadata],
    ) -> None:
        """Append the auxdata datasets for the current iteration."""
        # Check if there are any simulations
        if not cur_sims:
            return
        # Create the auxdata datasets for the current iteration
        for name in cur_sims[0].auxdata:
            # Concatenate the auxdata from all the simulations
            auxdata = np.array([x.auxdata[name] for x in cur_sims])

            # Create the dataset
            iter_group.create_dataset(f'auxdata/{name}', data=auxdata)

    def append(
        self,
        cur_sims: list[SimMetadata],
        basis_states: BasisStates,
        target_states: list[TargetState],
        metadata: IterationMetadata,
    ) -> None:
        """Append the next iteration to the HDF5 file."""
        # Get the current iteration number (1-indexed)
        n_iter = metadata.iteration_id

        with h5py.File(self.westpa_h5file_path, mode='a') as f:
            # Update the attrs for the current iteration
            f.attrs['west_current_iteration'] = n_iter

            # Append the summary table row
            self._append_summary(f, n_iter, cur_sims, metadata)

            # Append the basis states if we are on the first iteration
            if n_iter == 1:
                self._append_ibstates(f, n_iter, basis_states)

            # Append the target states if we are on the first iteration
            if n_iter == 1:
                self._append_tstates(f, n_iter, target_states)

            # Append the bin mapper if we are on the first iteration
            # NOTE: this assumes the binning scheme does not change.
            if n_iter == 1:
                self._append_bin_mapper(f, metadata)

            # TODO: We may need to add istate_index, istate_pcoord into the
            #       ibstates group. But for now, we are not.

            # Create the iteration group
            iter_group: h5py.Group = f.require_group(
                '/iterations/iter_{:0{prec}d}'.format(
                    n_iter,
                    prec=self.west_iter_prec,
                ),
            )

            # Add the iteration number and binhash to the group attributes
            iter_group.attrs['n_iter'] = n_iter
            iter_group.attrs['binhash'] = metadata.binner_hash

            # Append the seg_index table
            self._append_seg_index_table(iter_group, cur_sims)

            # Append the pcoords
            self._append_pcoords(iter_group, cur_sims)

            # Append the bin_target_counts
            self._append_bin_target_counts(iter_group, metadata)

            # Append the ibstates datasets for the current iteration
            self._append_iter_ibstates(f, iter_group, n_iter)

            # Append the tstates datasets for the current iteration
            self._append_iter_tstates(f, iter_group, n_iter)

            # Append the auxdata datasets for the current iteration
            self._append_auxdata(iter_group, cur_sims)
