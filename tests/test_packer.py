import torch

from navit_rf.navit import make_packing_collate


def test_packing_collate_shapes():
    collate = make_packing_collate(patch_size=4, max_tokens_per_pack=32)
    imgs = [torch.rand(3, 10, 12), torch.rand(3, 6, 6)]
    batch = collate(imgs)
    assert batch["images"].shape[0] == 2
    assert batch["patch_hw"].shape == (2, 2)
    assert isinstance(batch["packs"], list)
