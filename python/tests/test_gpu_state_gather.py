import torch

from kangengine.state import GPUStateBackend


class _MetadataBackend:
    def __init__(self, num_envs):
        self._complete_obj_ids = {0}
        self._registered_env_ids = {0: set(range(num_envs))}
        self.calls = 0

    def record(self, env_id, obj_id):
        self.calls += 1
        return env_id, obj_id


def test_persistent_index_select_reuses_output_buffer():
    backend = GPUStateBackend(num_envs=2, device="cpu")
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    indices = torch.tensor([2, 0], dtype=torch.long)

    first = backend._persistent_index_select(source, 1, indices, "dofs")
    first_ptr = first.data_ptr()

    source.add_(100.0)
    second = backend._persistent_index_select(source, 1, indices, "dofs")

    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, source[:, indices])


def test_row_and_column_selection_reuses_both_gather_buffers():
    backend = GPUStateBackend(num_envs=2, device="cpu")
    source = torch.arange(60, dtype=torch.float32).reshape(3, 5, 4)
    rows = torch.tensor([2, 0], dtype=torch.long)
    columns = torch.tensor([3, 1], dtype=torch.long)

    first = backend._select_rows_and_columns(
        source,
        rows,
        None,
        columns,
        None,
        "links",
    )
    first_ptr = first.data_ptr()

    source.mul_(2.0)
    second = backend._select_rows_and_columns(
        source,
        rows,
        None,
        columns,
        None,
        "links",
    )

    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, source[rows][:, columns])


def test_object_records_are_cached_after_registration_is_complete():
    backend = GPUStateBackend(num_envs=4, device="cpu")
    metadata = _MetadataBackend(backend.num_envs)
    backend.set_metadata_backend(metadata)

    first = backend._records_for_obj(0)
    second = backend._records_for_obj(0)

    assert first is second
    assert metadata.calls == backend.num_envs
