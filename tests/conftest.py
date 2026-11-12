import pytest

from valta.chat import wait_for_facts


@pytest.fixture(autouse=True)
def _facts_are_read_before_the_next_test():
    yield
    wait_for_facts()
