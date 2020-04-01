from typing import Union, Sequence, Optional
from colorama import Fore, Back
import torch
import math


class TensorAssertionError(AssertionError):
    pass


def print_as_error_intro(string: str):
    print(Fore.BLACK + Back.RED +
          string +
          Back.RESET + Fore.RESET)


class AssertTensor:
    def __init__(self, actual_to_test: torch.Tensor):
        if not isinstance(actual_to_test, torch.Tensor):
            raise ValueError(
                f"The actual to test should be a tensor. Instead got {type(actual_to_test)}"
            )
        self.actual = actual_to_test

    def is_close_to(self, expected: Union[torch.Tensor, Sequence], epsilon=1e-7) -> None:
        if isinstance(expected, torch.Tensor):
            if expected.device != self.actual.device:
                print_as_error_intro(
                    f"Tensors not on same device. Actual is {self.actual.device}. Expect {expected.device}")
                raise TensorAssertionError()
        if isinstance(self.actual, torch.LongTensor):
            epsilon = math.ceil(epsilon)
        if isinstance(expected, list):
            if isinstance(self.actual, torch.LongTensor):
                expected = torch.LongTensor(expected)
            else:
                expected = torch.tensor(expected)
        if expected.shape != self.actual.shape:
            print_as_error_intro(
                  "The two tensors do not have the same shape.")
            should_print_full_tensor = self.actual.numel() < 20
            print(f"Actual Shape: {self.actual.shape}")
            if should_print_full_tensor:
                print(f"  {self.actual}")
            print(f"Expected Shape: {expected.shape}")
            if should_print_full_tensor:
                print(f"  {expected}")
            # TODO check to see if an unsqueeze away
            raise TensorAssertionError()
        expected = expected.to(self.actual.device)

        difs = None
        if self.actual.dtype == torch.bool and expected.dtype == torch.bool:
            difs = self.actual != expected
            if not torch.any(difs):
                return
        else:
            try:
                difs = torch.abs(torch.add(self.actual, -expected))
            except RuntimeError as e:
                print(e)
            else:
                eqs = torch.lt(difs, epsilon)
                if torch.all(eqs):
                    return

        print("Epsilon equals failure.")
        print(f"Expected:\n {expected}")
        print(f"Got:\n {self.actual}")
        if difs is not None:
            print(f"Difs:\n {difs}")
        raise TensorAssertionError()

    def has_shape(self, *expected: Optional[int]):
        def fail():
            print_as_error_intro(
                "The two tensors do not have the same shape.")
            print(f"Actual Shape: {self.actual.shape}")
            print(f"Expected Shape: {expected}")
            raise TensorAssertionError()

        # A None at the end of expected means any extra shape is allowed.
        # if that is not the case then the shape of the shapes should be the same.
        if expected[-1] is not None and len(expected) != len(self.actual.shape):
            fail()

        # Go through and check each non-None dim.
        for i, exp in enumerate(expected):
            if exp is not None and exp != self.actual.shape[i]:
                fail()
