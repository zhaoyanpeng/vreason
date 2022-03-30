import copy
import logging
import types
from typing import List

__all__ = ["DatasetCatalog", "MetadataCatalog", "Metadata"]

class DatasetCatalog(object):
    _REGISTERED = {}
    _CALLED = {}

    @staticmethod
    def register(name, func, override=False):
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert override or name not in DatasetCatalog._REGISTERED, "Dataset '{}' is already registered!".format(
            name
        )
        DatasetCatalog._REGISTERED[name] = func
        if override and name in DatasetCatalog._REGISTERED:
            DatasetCatalog._CALLED.pop(name, None)

    @staticmethod
    def get(name):
        try:
            f = DatasetCatalog._REGISTERED[name]
        except KeyError:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(DatasetCatalog._REGISTERED.keys())
                )
            )
        message = "calling"
        if name not in DatasetCatalog._CALLED: 
            DatasetCatalog._CALLED[name] = f()
            message += " is being processed"
        return DatasetCatalog._CALLED[name]

    @staticmethod
    def list() -> List[str]:
        return list(DatasetCatalog._REGISTERED.keys())

    @staticmethod
    def clear():
        DatasetCatalog._REGISTERED.clear()
        DatasetCatalog._CALLED.clear()

    @staticmethod
    def remove(name):
        DatasetCatalog._REGISTERED.pop(name, None)
        DatasetCatalog._CALLED.pop(name, None)

class Metadata(types.SimpleNamespace):
    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            return getattr(self, self._RENAMED[key])

        # "name" exists in every metadata
        if len(self.__dict__) > 1:
            raise AttributeError(
                "Attribute '{}' does not exist in the metadata of dataset '{}'. Available "
                "keys are {}.".format(key, self.name, str(self.__dict__.keys()))
            )
        else:
            raise AttributeError(
                f"Attribute '{key}' does not exist in the metadata of dataset '{self.name}': "
                "metadata is empty."
            )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except AttributeError:
            return default

class MetadataCatalog:
    _NAME_TO_META = {}

    @staticmethod
    def get(name):
        assert len(name)
        if name in MetadataCatalog._NAME_TO_META:
            return MetadataCatalog._NAME_TO_META[name]
        else:
            m = MetadataCatalog._NAME_TO_META[name] = Metadata(name=name)
            return m

    @staticmethod
    def list():
        return list(MetadataCatalog._NAME_TO_META.keys())

    @staticmethod
    def clear():
        MetadataCatalog._NAME_TO_META.clear()

    @staticmethod
    def remove(name):
        MetadataCatalog._NAME_TO_META.pop(name)
