This is simple WIP hacky testing/assertion/logging library for Pytorch stuff.

It was mostly intended for personal use. As I continue add to it I'll hopefully
eventually clean it up so others could more easily use.

## Nice assertions on tensors

Combustable provides utilites for making assertions
on tensors and better outputs on failure.

```Python
import torch
from combustable.equality import AssertTensor

a = torch.tensor([[1.0, 2, 3]])

# Make assertions on our tensor
AssertTensor(a).is_close_to(torch.tensor([[1.0, 2, 3]]))
# List-like's are cast into tensors for you. More concise.
AssertTensor(a).is_close_to([[1, 2, 3]])
# Depending on what you're doing might need to adjust epsilon
AssertTensor(a).is_close_to([[1.01, 2, 3]], epsilon=1e-1)
# Other kinds of assertions
AssertTensor(a).has_shape(1, 3)
```

The autocasting when comparing with lists was one of the main motivation for
Combustable. Right now Combustable's output on failure is ok (better than a naked assertion, at least), 
but also want to eventually make it much more pretty/debug-friendly.

## Assertions on assign.

A common idiom in pytorch is unpacking the shape of something
and then verify the shape is as expected. 

Combustable introduces some syntactic sugar for doing this more concisely.

```Python
# Without this sugar
def my_cool_layer_func(a: torch.Tensor, b.torch.Tensor):
    batch_size, seq_len, hidden = a.shape
    b_batch_size, class_ind = b.shape
    assert batch_size == b_batch_size
    # Having these variables unpacked is nice as it is self documenting
    # and gives us access to the axis-size info if we need them in the method.
    # However, it'd be nice if we didn't have to have this `b_batch_size` temp var.
    # We only really expect one batch_size.

# With assertions on assign
from combustable.assert_on_assign import eq
def my_cool_layer_func(a: torch.Tensor, b.torch.Tensor):
    batch_size, seq_len, hidden = a.shape
    eq[batch_size], class_ind = b.shape
    # If the first dim of `b` != batch_size we will get an error.
    # The assertion is automatic. This is concise and avoids polluting our namespace
    # with a temp variable.
```

# Install

```bash
pip install git+https://github.com/DNGros/combustable.git
# OR
git clone https://github.com/DNGros/combustable.git
cd combustable
pip install .
```
Like I mentioned, it's mostly WIP/for personal use. So no pypi package yet.
