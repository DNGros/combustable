from typing import TypeVar

T = TypeVar('T')


class AssignEqClass:
    """Allows for assertions done during assignment/unpacking
    
    Examples:
        batch_size, seq_len, hidden = a.shape
        eq[batch_size], class_ind = b.shape
    """
    def __setitem__(self, expect: T, got: T):
        assert expect == got, f"Expect: {expect}. Got {got}"

    def __getitem__(self, item):
        raise ValueError("You don't actually get from this, just assign")


eq = AssignEqClass()
