## Copyright (C) 2023 Philipp Benner

import collections

## ----------------------------------------------------------------------------


class ModelConfig(collections.UserDict):
    def __init__(self, _model_config):
        super().__init__(self)
        for key, value in _model_config.items():
            super().__setitem__(key, value)

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

        return self

    def __setitem__(self, key, value):
        # Check that we do not accept any invalid
        # config options
        if key in self:
            super().__setitem__(key, value)
        else:
            raise KeyError(key)

    def __str__(self):
        result = "Model config:\n"
        for key, value in self.items():
            result += f"-> {key:21}: {value}\n"
        return result

    def __copy__(self):
        return ModelConfig(self)

    def __deepcopy__(self, memo):
        result = ModelConfig(self)
        memo[id(self)] = result
        return result
