class _DummyScrollBar:
    def __init__(self, visible):
        self.visible = visible


class _Slot:
    def __init__(self):
        self.element = None


class ListBox2D:
    def __init__(self, values, *, multiselection=True):
        self.values = list(values)
        self.multiselection = multiselection
        self.selected = []
        self.last_selection_idx = None

        self.nb_slots = len(self.values)
        self.slots = [_Slot() for _ in self.values]

        self.scroll_bar = _DummyScrollBar(
            visible=len(self.values) > self.nb_slots
        )

    def select(self, item, *, multiselect=False, range_select=False):
        if item.element not in self.values:
            return

        selection_idx = self.values.index(item.element)

        if range_select and self.last_selection_idx is not None:
            start = min(self.last_selection_idx, selection_idx)
            end = max(self.last_selection_idx, selection_idx)
            self.selected = self.values[start : end + 1]

        elif self.multiselection and multiselect:
            if item.element in self.selected:
                self.selected.remove(item.element)
            else:
                self.selected.append(item.element)
            self.last_selection_idx = selection_idx

        else:
            self.selected = [item.element]
            self.last_selection_idx = selection_idx
