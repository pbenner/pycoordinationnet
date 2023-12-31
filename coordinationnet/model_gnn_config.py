## Copyright (C) 2023 Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## ----------------------------------------------------------------------------

from .model_config import ModelConfig

## ----------------------------------------------------------------------------

_graph_config = {
    'num_convs'     : 2,
    'conv_type'     : 'ResGatedGraphConv',
    'rbf_type'      : 'Gaussian',
    'dim_element'   : 200,
    'dim_oxidation' : 10,
    'dim_geometry'  : 10,
    'dim_csm'       : 128,
    'dim_distance'  : 128,
    'dim_angle'     : 128,
    'bins_csm'      : 20,
    'bins_distance' : 20,
    'bins_angle'    : 20,
    'oxidations'    : True,
    'distances'     : True,
    'geometries'    : True,
    'csms'          : True,
    'angles'        : True,
}

## ----------------------------------------------------------------------------

GraphCoordinationNetConfig = ModelConfig(_graph_config)
