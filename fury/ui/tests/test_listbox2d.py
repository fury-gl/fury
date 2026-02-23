import numpy as np
from fury.ui import ListBox2D


def test_listbox_creation():
    lb = ListBox2D(values=["a", "b", "c"])
    assert lb.values == ["a", "b", "c"]
    assert lb.selected == []


def test_single_selection():
    lb = ListBox2D(values=["a", "b", "c"])
    item = lb.slots[0]
    item.element = "a"
    lb.select(item)
    assert lb.selected == ["a"]


def test_multiselect():
    lb = ListBox2D(values=["a", "b", "c"], multiselection=True)
    item1 = lb.slots[0]
    item1.element = "a"
    lb.select(item1, multiselect=True)

    item2 = lb.slots[1]
    item2.element = "b"
    lb.select(item2, multiselect=True)

    assert set(lb.selected) == {"a", "b"}


def test_range_select():
    lb = ListBox2D(values=["a", "b", "c", "d"], multiselection=True)

    item1 = lb.slots[0]
    item1.element = "a"
    lb.select(item1)

    item2 = lb.slots[2]
    item2.element = "c"
    lb.select(item2, range_select=True)

    assert lb.selected == ["a", "b", "c"]


def test_scrollbar_hidden_when_not_needed():
    lb = ListBox2D(values=["a"])
    assert lb.scroll_bar.visible is False or lb.nb_slots >= len(lb.values)
