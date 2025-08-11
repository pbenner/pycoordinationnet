## Copyright (C) 2023 Philipp Benner

import numpy as np

from coordinationnet import mp_icsd_query, mp_icsd_clean

# %% Retrieve oxides from materials project
### ---------------------------------------------------------------------------

mats = mp_icsd_query("Q0tUKnAE52sy7hVO", experimental_data=False)
mats = mp_icsd_clean(mats)
# Remove 'mp-554015' due to bug #2756
mats = np.delete(mats, 3394)
# Need to convert numpy array to list for serialization
mats = mats.tolist()

# %% Extract structures and target values, convert structures to
### coordination features (this may take a while)
### ---------------------------------------------------------------------------

from coordinationnet import CoordinationFeaturesData

structures = [mat["structure"] for mat in mats]
targets = [[mat["formation_energy_per_atom"]] for mat in mats]

data = CoordinationFeaturesData(structures, y=targets, verbose=True)

# %% Save data
### ---------------------------------------------------------------------------

data.save("mpoxides.dill")

# %% Load data
### ---------------------------------------------------------------------------

data = CoordinationFeaturesData.load("mpoxides.dill")

# %%
