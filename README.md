## CoordinationNet

CoordinationNet is a transformer model that uses coordination information to predict materials properties. It is implemented in pytorch/lightning and provides a simple interface for training, predicting and cross-validation.

### Model initialization

The following code creates a new model instance with default settings:
```python
from coordinationnet import CoordinationNet

model = CoordinationNet()
```

### Creating a data set

As an example, we retrieve experimental and theoretical oxides from the materials project
```python
import numpy as np

from coordinationnet import mp_icsd_query, mp_icsd_clean

mats = mp_icsd_query("Q0tUKnAE52sy7hVO", experimental_data = False)
mats = mp_icsd_clean(mats)
# Remove 'mp-554015' due to bug #2756
mats = np.delete(mats, 3394)
# Need to convert numpy array to list for serialization
mats = mats.tolist()
```

We extract structures and target values (formation energy) from our materials list:
```python
structures = [  mat['structure']                  for mat in mats ]
targets    = [ [mat['formation_energy_per_atom']] for mat in mats ]
```

For training the network or making predictions, we must convert structures to cooridnation features. This is achieved using:
```python
from coordinationnet import CoordinationFeaturesData

data = CoordinationFeaturesData(structures, y = targets, verbose = True)
```
Note that *y* (i.e. the target values) is optional and can be left empty if the network is not trained and only used for making predictions on this data.

### Run cross-validation
CoordinationNet implements a cross-validation method that can be easily used:
```python
from monty.serialization import dumpfn

mae, y, y_hat = model.cross_validation(data, 10)

print('Final MAE:', mae)

# Save result
dumpfn({'y_hat': y_hat.tolist(),
        'y'    : y    .tolist(),
        'mae'  : mae },
        'eval-test.txt')
```

### Train and predict
Model training and computing predictions:
```python
import matplotlib.pyplot as plt

result = model.train(data)

plt.plot(result['train_error'])
plt.plot(result['val_error'])
plt.show()

model.predict(data)
```

### Save and load model
A trained model can be easily saved and loaded using:
```python
model.save('model.dill')
model = CoordinationNet.load('model.dill')
```

---
---
### Advanced model initialization
CoordinationNet has a modular structure so that individual components can be included or excluded from the model. Which model components are included is controlled using the *CoordinationNetConfig*. The following shows the default values for CoordinationNet:
```python
from coordinationnet import CoordinationNetConfig

model_config = CoordinationNetConfig(
    composition           = False,
    sites                 = False,
    sites_oxid            = False,
    sites_ces             = False,
    site_features         = True,
    site_features_ces     = True,
    site_features_oxid    = True,
    site_features_csms    = True,
    site_features_ligands = False,
    ligands               = False,
    ce_neighbors          = False,
)
```
If a value is not specified, it will be set to *False* by default. Hence, the above configuration is equivalen to
```python
model_config = CoordinationNetConfig(
    site_features         = True,
    site_features_ces     = True,
    site_features_oxid    = True,
    site_features_csms    = True,
)
```

The following code creates a new model instance with additional keyword arguments:
```python
from coordinationnet import CoordinationNet

model = CoordinationNet(
    # Model components
    model_config = model_config,
    # Dense layer options
    layers = [200, 4096, 1024, 512, 128, 1], dropout = 0.0, skip_connections = False, batchnorm = False,
    # Transformer options
    edim = 200, nencoders = 4, nheads = 4, dropout_transformer = 0.0, dim_feedforward = 200,
    # Data options
    batch_size = 128, num_workers = 10,
    # Optimizer options
    scheduler = 'plateau', devices=[2], patience = 2, lr = 1e-4, max_epochs = 1000)

```
All keyword arguments correspond to default values and don't have to be specified unless changed.

---
---
## Coordination Features

This packages uses *pymatgen* to compute coordination environments, their distances and angles. The output is such that it can be used as features for machine learning models.

A coordination environment consists of a central atom or ion, which is usually metallic and is called the coordination center. It is surrounded by several bounding ions that are called ligands. The position of ligands defines the shape of the coordination environment, which is described in terms of the coordination number (number of ligands) and a symbol for the shape of the environment (e.g. C for cubic)

### Getting started

First, we download a material from the materials project:
```python
from pymatgen.ext.matproj import MPRester

mid = 'mp-12236'

with MPRester("Q0tUKnAE52sy7hVO") as m:
    structure = m.get_structure_by_material_id(mid, conventional_unit_cell=True)
```

The crystal features can be computed directly from the structure object:
```python
from coordinationnet import CoordinationFeatures

features = CoordinationFeatures.from_structure(structure)
```

The *features* object contains information about the oxidation state of sites (*oxidation*), the local environments (*ce*), the nearest neighbor distances (*distances*), the distances to neighboring coordination environments (*ce_distances*), and the angles between coordination environments (*ce_angles*). Note that a site may have multiple local environments.

### Elements, oxidations, ions, and distances

The *sites* substructure contains basic information about each site, including elements, oxidation states, and the type of ion. For instance, the oxidation states can be accessed using:
```python
>>> features.sites.oxidations
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
```
The list contains the oxidation states for all 160 sites in the material.

The distances between cations neighboring ligands is stored in the *distances* item:
```python
>>> features.distances[0]
{'site': 0, 'site_to': 102, 'distance': 2.341961770123179}
```
We see that the first item contains the distance between site 0 (cation) and site 102 (ligand).

### Coordination environments

The *ces* substructure contains all coordination environments of the material. The first set of environments can be accessed using:
```python
>>> features.ces[0]
{'site': 0, 'ce_symbols': ['C:8'], 'ce_fractions': [1.0], 'csms': [2.217881949143581], 'permutations': [[2, 4, 0, 1, 7, 5, 3, 6]]}
```
We see that this set of environments belongs to site 0. The *ce_symbols* item lists the symbol of the coordination environments (*ce*) as defined in the supplementary material of *Waroquiers et al. 2020* [1]. In this example we have *C:8* which refers to a cube where the central site has 8 neighbors. Each coordination environment is attributed a fraction given by the *ce_fractions* item. Since we have only a single coordination environment in this example, the *ce_fractions* item contains a single value of one. The Continuous Symmetry Measure (*CSM*) specifies the distance of the coordination environment to the perfect model environment (given by the *csms* item) [2]. The CSM value ranges between zero and 100, where a value of zero represents a perfect match. To compute the similarity of the coordination environment, all possible permutations of neighboring sites must be tested. The permutation with the minimal CSM is given by the *permutation* item.

Note that coordination environments are only computed for cations, because they form the centers of coordination environments.

### Neighboring coordination environments

The *ce_neighbors* item contains a list of all neighboring coordination environments. Note that neighbors are computed by considering symmetries stemming from the periodic boundary conditions. The first item can be accessed using:
```python
>>> features.ce_neighbors[0]
{'site': 0, 'site_to': 2, 'distance': 3.781941127654686, 'connectivity': 'edge', 'ligand_indices': [93, 94], 'angles': [104.76176533545028, 104.76176533545029]}
```
We see that it specifies the distance between sites 0 and 1, two cations that form the centers of two coordination environments. The environments are connected through an edge, hence it has two ligands. For both ligands, we also obtain the angle measured at the ligand.

### Encoded features

Features contain several categorical variables, for instance element names type of angles between coordination environments (i.e. corner, face, etc.). These variables must be encoded such that they can be used as inputs to machine learning models. The simples approach is to replace categorical varibales by integer indices, which enables us to use embeddings for categorical variables. Also the value of oxidation states is not very well suited for machine learning models, which are positive and negative integer values. We also recode oxidation states as positive integers, such that embeddings can be used.

Features can be encoded and decoded as follows:
```python
features = features.encode()
features = features.decode()
```

## References

[1] Waroquiers, David, et al. "ChemEnv: a fast and robust coordination environment identification tool." Acta Crystallographica Section B: Structural Science, Crystal Engineering and Materials 76.4 (2020): 683-695.

[2] Pinsky, Mark, and David Avnir. "Continuous symmetry measures. 5. The classical polyhedra." Inorganic chemistry 37.21 (1998): 5575-5582.
