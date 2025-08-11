## Copyright (C) 2023 Philipp Benner

import torch

from .features_coding import NumOxidations, NumGeometries

from .model_layers import (
    TorchStandardScaler,
    ModelDense,
    ElementEmbedder,
    RBFLayer,
    AngleLayer,
)
from .model_transformer_config import DefaultTransformerCoordinationNetConfig

## ----------------------------------------------------------------------------


class ModelComposition(torch.nn.Module):
    def __init__(self, edim, **kwargs):
        super().__init__()

        self.embedding_element = ElementEmbedder(edim, from_pretrained=True)

    def forward(self, x_comp):
        x = self.embedding_element(x_comp.elements)
        # x[x_comp.elements == NumElements] = 0.0
        # Sum over elements
        x = torch.sum(x, dim=1)
        # Normalize by number of elements
        x = torch.div(x.T, x_comp.sizes).T
        return x


## ----------------------------------------------------------------------------


class ModelSitesTransformer(torch.nn.Module):
    def __init__(
        self,
        # Encoder options
        edim,
        nheads=4,
        nencoders=4,
        dim_feedforward=2048,
        dropout_transformer=0.1,
        # Component options
        oxidation=True,
        ces=True,
        # Dense network options
        activation=torch.nn.ELU(),
        **kwargs,
    ):
        super().__init__()

        dim_input = edim
        dim_output = edim
        if oxidation:
            dim_input += edim
        if ces:
            dim_input += edim

        encoder_layer = torch.nn.TransformerEncoderLayer(
            dim_input,
            batch_first=True,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout_transformer,
        )

        self.transformer = torch.nn.TransformerEncoder(encoder_layer, nencoders)
        self.embedding_cls = torch.nn.Embedding(1, dim_input)
        self.embedding_element = ElementEmbedder(edim, from_pretrained=True)
        self.embedding_oxidation = None
        self.embedding_ces = None
        self.dense = ModelDense(
            [dim_input, dim_output], activation=activation, **kwargs
        )

        if oxidation:
            self.embedding_oxidation = torch.nn.Embedding(NumOxidations + 1, edim)

        if ces:
            self.embedding_ces = torch.nn.Embedding(NumGeometries + 1, edim)

    def forward(self, x):
        m = x.mask
        # Get element embedding
        embeddings = self.embedding_element(x.elements)
        # Concatenate optional features (element-wise)
        if self.embedding_oxidation is not None:
            embeddings = torch.cat(
                (embeddings, self.embedding_oxidation(x.oxidations)), dim=2
            )
        if self.embedding_ces is not None:
            embeddings = torch.cat((embeddings, self.embedding_ces(x.ces)), dim=2)
        # Apply transformer
        x = torch.cat((self.embedding_cls(x.cls), embeddings), dim=1)
        # Dimension of x is now:
        # (batch, sequence, edim)
        x = self.transformer(x, src_key_padding_mask=m)
        # Follow the BERT architecture and extract only the
        # first sequence element (cls) after applying the transformer
        x = x[:, 0, :]
        # Dimension of x is now:
        # (batch, edim)
        x = self.dense(x)

        return x


## ----------------------------------------------------------------------------


class ModelSiteLigandsTransformer(torch.nn.Module):
    def __init__(
        self,
        # Encoder options
        edim,
        nheads=4,
        nencoders=4,
        dim_feedforward=2048,
        dropout_transformer=0.1,
        # Dense network options
        activation=torch.nn.ELU(),
        **kwargs,
    ):
        super().__init__()

        nencoders = 1
        nheads = 4

        encoder_layer = torch.nn.TransformerEncoderLayer(
            edim,
            batch_first=True,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout_transformer,
        )

        self.transformer = torch.nn.TransformerEncoder(encoder_layer, nencoders)
        self.embedding_cls = torch.nn.Embedding(1, edim)
        self.embedding_element = ElementEmbedder(
            edim, from_pretrained=True, freeze=False
        )
        self.embedding_ligelem = ElementEmbedder(
            edim, from_pretrained=True, freeze=False
        )
        self.embedding_ligoxid = torch.nn.Embedding(NumOxidations + 1, edim)
        # self.rbf_angles        = RBFLayer(0, 180, dim = int(edim/2), dim_out = edim)
        # self.rbf_angles        = RBFLayer(0, 180, dim = 2048, dim_out = edim)
        self.dense_angles = AngleLayer(edim, [edim, edim], **kwargs)

    def forward(self, x_input):
        s = x_input.summation
        # Sum up the two element columns
        y = self.embedding_element(x_input.elements).sum(dim=1, keepdim=True)
        # Apply transformer to full data
        x = torch.cat(
            (
                self.embedding_cls(x_input.cls),
                self.embedding_ligelem(x_input.ligelem),
                self.embedding_ligoxid(x_input.ligoxid),
                #                self.rbf_angles       (x.angles[:,:,None]),
                # Attach sum of element embeddings
                y,
            ),
            dim=1,
        )
        # Dimension of x is now:
        # (batch, sequence, edim)
        x = self.transformer(x)
        # Follow the BERT architecture and extract only the
        # first sequence element (cls) after applying the transformer
        x = x[:, 0, :]
        x = self.dense_angles(x, x_input.distances, x_input.angles)
        # Dimension of x is now:
        # (batch, edim)
        # Each site has multiple CE-pairs and ligands connecting them, we
        # have to sum over all entries that belong to the same site. The
        # batch size then corresponds to the number of sites in the batch
        x = s.T @ x
        return x


## ----------------------------------------------------------------------------


class ModelSiteFeaturesTransformer(torch.nn.Module):
    def __init__(
        self,
        # Encoder options
        edim,
        transformer=True,
        nheads=4,
        nencoders=4,
        dim_feedforward=2048,
        dropout_transformer=0.1,
        # Component options
        oxidation=True,
        csms=True,
        ligands=True,
        # Dense network options
        activation=torch.nn.ELU(),
        **kwargs,
    ):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            edim,
            batch_first=True,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout_transformer,
        )

        self.csms = csms
        self.transformer = None
        self.transformer_ligands = None
        self.embedding_cls = torch.nn.Embedding(1, edim)
        self.embedding_element = ElementEmbedder(
            edim, from_pretrained=True, freeze=False
        )
        self.embedding_ces = torch.nn.Embedding(NumGeometries + 1, edim)
        self.embedding_oxidation = None

        if csms:
            # Dense layer for combining ce_symbols and csms
            self.dense = ModelDense(
                [edim + 1, edim], skip_connections=False, batchnorm=False
            )
        else:
            # We don't need a dense layer in this case, but want to ensure equal model capacity
            self.dense = ModelDense(
                [edim + 0, edim], skip_connections=False, batchnorm=False
            )

        if oxidation:
            self.embedding_oxidation = torch.nn.Embedding(NumOxidations, edim)

        if transformer:
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, nencoders)

        if ligands:
            self.transformer_ligands = ModelSiteLigandsTransformer(
                edim,
                nheads=nheads,
                nencoders=nencoders,
                dim_feedforward=dim_feedforward,
            )
            self.dense_ligands = torch.nn.Linear(2 * edim, edim)

    def forward_ces(self, x_ces):
        # Compute CE embeddings per site
        x = self.embedding_ces(x_ces.ce_symbols)
        # Add csms information if available
        if self.csms:
            x = torch.cat((x, x_ces.csms[:, None, :]), dim=2)
        # Reduce dimension to edim
        x = self.dense(x)
        x = x[:, 0, :]
        # Size of y is: (batch, 1, edim); map result to sites
        x = x_ces.summation.T @ x
        return x[:, None, :]

    def forward_ligands(self, x_ligands):
        x = self.transformer_ligands(x_ligands)
        return x[:, None, :]

    def forward(self, x_sites, x_ces, x_ligands):
        # Get element embeddings
        x = self.embedding_element(x_sites.elements)
        # Add optional features
        if self.embedding_oxidation is not None:
            x = torch.cat((x, self.embedding_oxidation(x_sites.oxidations)), dim=1)
        if x_ces is not None:
            x = torch.cat((x, self.forward_ces(x_ces)), dim=1)
        # Dimension of x is now:
        # (batch, sequence, edim)
        if self.transformer is not None:
            x = torch.cat((self.embedding_cls(x_sites.cls), x), dim=1)
            x = self.transformer(x)
            # Follow the BERT architecture and extract only the
            # first sequence element (cls) after applying the transformer
            x = x[:, 0, :]
        else:
            x = x.sum(dim=1)
        # Dimension of x is now:
        # (batch, edim)

        if self.transformer_ligands is not None:
            x_ligands = self.forward_ligands(x_ligands)[:, 0, :]
            x = self.dense_ligands(torch.cat((x, x_ligands), dim=1))

        # Each material has multiple sites, we have to sum over all entries
        # that belong to the same material. The batch size then corresponds
        # to the number of materials in the batch
        x = x_sites.summation.T @ x

        return x


## ----------------------------------------------------------------------------


class ModelLigandsTransformer(torch.nn.Module):
    def __init__(
        self,
        # Encoder options
        edim,
        nheads=4,
        nencoders=4,
        dim_feedforward=2048,
        dropout_transformer=0.1,
        # Dense network options
        activation=torch.nn.ELU(),
        **kwargs,
    ):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            edim,
            batch_first=True,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout_transformer,
        )

        self.transformer = torch.nn.TransformerEncoder(encoder_layer, nencoders)
        self.embedding_cls = torch.nn.Embedding(1, edim)
        self.embedding_element = ElementEmbedder(edim, from_pretrained=True)
        self.embedding_ligands = ElementEmbedder(edim, from_pretrained=True)
        # TODO:
        # Include distances between cations and ligands (stored in CoordinationFeatures.distances)
        self.rbf_angles = RBFLayer(0, 180, edim)

    def forward(self, x):
        s = x.summation
        # Sum up the two element columns
        y = self.embedding_element(x.elements).sum(dim=1, keepdim=True)
        # Apply transformer to full data
        x = torch.cat(
            (
                self.embedding_cls(x.cls),
                self.embedding_ligands(x.ligands),
                self.rbf_angles(x.angles[:, :, None]),
                # Attach CLS result from previous transformer
                y,
            ),
            dim=1,
        )
        # Dimension of x is now:
        # (batch, sequence, edim)
        x = self.transformer(x)
        # Follow the BERT architecture and extract only the
        # first sequence element (cls) after applying the transformer
        x = x[:, 0, :]
        # Dimension of x is now:
        # (batch, edim)
        # Each material has multiple sites, we have to sum over all entries
        # that belong to the same material. The batch size then corresponds
        # to the number of materials in the batch
        x = s.T @ x
        return x


## ----------------------------------------------------------------------------


class ModelCeNeighborsTransformer(torch.nn.Module):
    def __init__(
        self,
        # Encoder options
        edim,
        nheads=4,
        nencoders=4,
        dim_feedforward=2048,
        dropout_transformer=0.1,
        # Component options
        transformer_element=False,
        transformer_ces=False,
        transformer=True,
        # Dense network options
        activation=torch.nn.ELU(),
        **kwargs,
    ):
        super().__init__()

        encoder_layer = torch.nn.TransformerEncoderLayer(
            edim,
            batch_first=True,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout_transformer,
        )

        self.transformer_element = None
        self.transformer_ces = None
        self.transformer = None

        if transformer_element:
            self.embedding_cls1 = torch.nn.Embedding(1, edim)
            self.transformer_element = torch.nn.TransformerEncoder(
                encoder_layer, nencoders
            )
        if transformer_ces:
            self.embedding_cls2 = torch.nn.Embedding(1, edim)
            self.transformer_ces = torch.nn.TransformerEncoder(encoder_layer, nencoders)
        if transformer:
            self.embedding_cls3 = torch.nn.Embedding(1, edim)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, nencoders)

        self.embedding_element = ElementEmbedder(edim, from_pretrained=True)
        self.embedding_ces = torch.nn.Embedding(NumGeometries + 1, edim)
        # self.embedding_connectivity = torch.nn.Embedding(NumAngleTypes+0, edim)
        # self.rbf_distances          = RBFLayer(0, 3, edim)

    def forward_element(self, x_input):
        if self.transformer_element is not None:
            y = torch.cat(
                (
                    self.embedding_cls1(x_input.cls),
                    self.embedding_element(x_input.elements),
                ),
                dim=1,
            )
            y = self.transformer_element(y)
            y = y[:, 0:1, :]
        else:
            y = self.embedding_element(x_input.elements)
            y = torch.sum(y, dim=1)
            y = y[:, None, :]
        return y

    def forward_site_ces(self, x_site_ces):
        # Obtain CE embeddings, each site can have multiple embeddings
        x = self.embedding_ces(x_site_ces.ce_symbols)
        # Size of y is: (ces, 1, edim); map result to sites, i.e. (sites, 1, edim)
        x = x_site_ces.summation.T @ x[:, 0, :]
        return x

    def forward_ces(self, x_input, x_site_ces):
        # Get CE information per site
        z = self.forward_site_ces(x_site_ces)
        # Map per site information to CE-pairs, i.e. for each CE-pair
        # get the CE of site and site_to (two columns per CE-pair)
        z = z[x_input.ce_index]
        if self.transformer_ces is not None:
            z = torch.cat((self.embedding_cls2(x_input.cls), z), dim=1)
            z = self.transformer_ces(z)
            z = z[:, 0:1, :]
        else:
            z = torch.sum(z, dim=1)
            z = z[:, None, :]
        return z

    def forward(self, x_input, x_ligands, x_site_ces):
        # Prepare data
        x = torch.cat(
            (
                # self.embedding_connectivity(x.connectivity),
                # self.rbf_distances         (x.distances[:,:,None]),
                self.forward_element(x_input),
                self.forward_ces(x_input, x_site_ces),
            ),
            dim=1,
        )
        # Add ligands to data if available
        if x_ligands is not None:
            x = torch.cat((x, x_ligands[:, None, :]), dim=1)
        # Dimension of x is now:
        # (batch, sequence, edim)
        if self.transformer is not None:
            x = torch.cat((self.embedding_cls3(x_input.cls), x), dim=1)
            x = self.transformer(x)
            # Follow the BERT architecture and extract only the
            # first sequence element (cls) after applying the transformer
            x = x[:, 0, :]
        else:
            x = torch.sum(x, dim=1)
        # Dimension of x is now:
        # (batch, edim)
        # Each material has CE-pairs, we have to sum over all entries
        # that belong to the same material. The batch size then corresponds
        # to the number of materials in the batch
        x = x_input.summation.T @ x
        return x


## ----------------------------------------------------------------------------


class ModelCoordinationNet(torch.nn.Module):
    def __init__(
        self,
        # Specify model components
        model_config=DefaultTransformerCoordinationNetConfig,
        # Transformer options
        edim=200,
        nencoders=4,
        nheads=4,
        dim_feedforward=200,
        dropout_transformer=0.0,
        # **kwargs contains options for dense layers
        layers=[200, 4096, 1024, 512, 128, 1],
        **kwargs,
    ):
        super().__init__()

        # The model config determines which components of the model
        # are active
        self.model_config = model_config
        # Optional scaler of model outputs (predictions)
        self.scaler_outputs = TorchStandardScaler(layers[-1])
        # Optional model components
        self.transformer_composition = None
        self.transformer_sites = None
        self.transformer_site_features = None
        self.transformer_ligands = None
        self.transformer_ce_neighbors = None

        # Optional site components
        sites_oxid = model_config["sites_oxid"]
        sites_ces = model_config["sites_ces"]
        # Optional site feature components
        site_features_oxid = model_config["site_features_oxid"]
        site_features_csms = model_config["site_features_csms"]
        site_features_ligands = model_config["site_features_ligands"]

        # Determine input dimension of the final dense neural network
        dim_dense_in = edim

        if model_config["composition"]:
            self.transformer_composition = ModelComposition(
                edim,
                nencoders=nencoders,
                nheads=nheads,
                dropout_transformer=dropout_transformer,
                dim_feedforward=dim_feedforward,
                **kwargs,
            )
        if model_config["sites"]:
            self.transformer_sites = ModelSitesTransformer(
                edim,
                nencoders=nencoders,
                nheads=nheads,
                dropout_transformer=dropout_transformer,
                dim_feedforward=dim_feedforward,
                oxidation=sites_oxid,
                ces=sites_ces,
                **kwargs,
            )
        if model_config["ligands"]:
            self.transformer_ligands = ModelLigandsTransformer(
                edim,
                nencoders=nencoders,
                nheads=nheads,
                dropout_transformer=dropout_transformer,
                dim_feedforward=dim_feedforward,
                **kwargs,
            )
        if model_config["site_features"]:
            self.transformer_site_features = ModelSiteFeaturesTransformer(
                edim,
                nencoders=nencoders,
                nheads=nheads,
                dropout_transformer=dropout_transformer,
                dim_feedforward=dim_feedforward,
                oxidation=site_features_oxid,
                csms=site_features_csms,
                ligands=site_features_ligands,
                **kwargs,
            )
        if model_config["ce_neighbors"]:
            self.transformer_ce_neighbors = ModelCeNeighborsTransformer(
                edim,
                nencoders=nencoders,
                nheads=nheads,
                dropout_transformer=dropout_transformer,
                dim_feedforward=dim_feedforward,
                **kwargs,
            )

        # Final dense layer
        self.dense = ModelDense([dim_dense_in] + layers, **kwargs)

    def _add_if_available_(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    def forward(self, x):
        x_composition = None
        x_sites = None
        x_site_features = None
        x_ligands = None
        x_ce_neighbors = None

        if self.transformer_composition is not None:
            x_composition = self.transformer_composition(x.composition)

        if self.transformer_sites is not None:
            x_sites = self.transformer_sites(x.sites)

        if self.transformer_site_features is not None:
            x_site_features = self.transformer_site_features(
                x.site_features, x.site_features_ces, x.site_features_ligands
            )

        if self.transformer_ligands is not None:
            x_ligands = self.transformer_ligands(x.ligands)

        if self.transformer_ce_neighbors is not None:
            x_ce_neighbors = self.transformer_ce_neighbors(
                x.ce_neighbors, x_ligands, x.site_features_ces
            )

        # Sum up all results
        x_input = None
        x_input = self._add_if_available_(x_input, x_composition)
        x_input = self._add_if_available_(x_input, x_sites)
        x_input = self._add_if_available_(x_input, x_site_features)
        x_input = self._add_if_available_(x_input, x_ce_neighbors)

        # Feed sum through final dense layer
        x = self.dense(x_input)
        x = self.scaler_outputs.inverse_transform(x)

        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters_grouped(self):
        parameters_linear_weight = []
        parameters_linear_bias = []
        parameters_transformer_weight = []
        parameters_transformer_bias = []
        parameters_transformer_linear_weight = []
        parameters_transformer_linear_bias = []
        parameters_embedding_weight = []
        parameters_embedding_bias = []
        parameters_norm_weight = []
        parameters_norm_bias = []
        parameters_other = []
        for name, param in self.named_parameters():
            # Norm layer parameters (this must come first, becasue we also collect transformer norm layers)
            if "norm" in name and "weight" in name:
                parameters_norm_weight.append(param)
            elif "norm" in name and "bias" in name:
                parameters_norm_bias.append(param)
            # Transformer parameters
            elif "transformer" in name:
                if "linear" in name and "weight" in name:
                    parameters_transformer_linear_weight.append(param)
                elif "linear" in name and "bias" in name:
                    parameters_transformer_linear_bias.append(param)
                elif "weight" in name:
                    parameters_transformer_weight.append(param)
                elif "bias" in name:
                    parameters_transformer_bias.append(param)
                else:
                    parameters_other.append(param)
            # Embedding parameters
            elif "embedding" in name and "weight" in name:
                parameters_embedding_weight.append(param)
            elif "embedding" in name and "bias" in name:
                parameters_embedding_bias.append(param)
            # Dense layer parameters
            elif "linear" in name and "weight" in name:
                parameters_linear_weight.append(param)
            elif "linear" in name and "bias" in name:
                parameters_linear_bias.append(param)
            else:
                parameters_other.append(param)

        return {
            "linear_weight": parameters_linear_weight,
            "linear_bias": parameters_linear_bias,
            "transformer_weight": parameters_transformer_weight,
            "transformer_bias": parameters_transformer_bias,
            "transformer_linear_weight": parameters_transformer_linear_weight,
            "transformer_linear_bias": parameters_transformer_linear_bias,
            "embedding_weight": parameters_embedding_weight,
            "embedding_bias": parameters_embedding_bias,
            "norm_weight": parameters_norm_weight,
            "norm_bias": parameters_norm_bias,
            "other": parameters_other,
        }
