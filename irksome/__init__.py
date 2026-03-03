from .tableaux.ButcherTableaux import (
    Alexander,
    BackwardEuler,
    GaussLegendre,
    LobattoIIIA,
    LobattoIIIC,
    PareschiRusso,
    QinZhang,
    RadauIIA,
)
from .tableaux.pep_explicit_rk import PEPRK
from .ufl.deriv import Dt, expand_time_derivatives
from .tableaux.dirk_imex_tableaux import DIRK_IMEX
from .tableaux.ars_dirk_imex_tableaux import ARS_DIRK_IMEX
from .tableaux.sspk_tableau import SSPK_DIRK_IMEX, SSPButcherTableau

from .constant import MeshConstant
from .tableaux.wso_dirk_tableaux import WSODIRK
from .scheme import create_time_quadrature
from .scheme import ContinuousPetrovGalerkinScheme, DiscontinuousGalerkinScheme
from .scheme import GalerkinCollocationScheme

__all__ = [
    "Alexander",
    "ARS_DIRK_IMEX",
    "BackwardEuler",
    "ContinuousPetrovGalerkinScheme",
    "create_time_quadrature",
    "DIRK_IMEX",
    "DIRKTimeStepper",
    "DiscontinuousGalerkinScheme",
    "Dt",
    "expand_time_derivatives",
    "GalerkinCollocationScheme",
    "GaussLegendre",
    "LobattoIIIA",
    "LobattoIIIC",
    "MeshConstant",
    "PareschiRusso",
    "PEPRK",
    "QinZhang",
    "RadauIIA",
    "SSPButcherTableau",
    "SSPK_DIRK_IMEX",
    "WSODIRK",
]


try:
    import importlib

    importlib.import_module("firedrake")
    from .bcs import BoundsConstrainedDirichletBC
    from .dirk_stepper import DIRKTimeStepper
    from .imex import RadauIIAIMEXMethod, DIRKIMEXMethod
    from .stage_derivative import getForm
    from .nystrom_dirk_stepper import DIRKNystromTimeStepper, ExplicitNystromTimeStepper
    from .nystrom_stepper import (
        StageDerivativeNystromTimeStepper,
        ClassicNystrom4Tableau,
    )
    from .stage_value import StageValueTimeStepper

    from .pc import (
        ClinesBase,
        ClinesLD,
        NystromAuxiliaryOperatorPC,
        RanaBase,
        RanaDU,
        RanaLD,
        IRKAuxiliaryOperatorPC,
    )
    from .galerkin_stepper import ContinuousPetrovGalerkinTimeStepper, TimeProjector
    from .discontinuous_galerkin_stepper import DiscontinuousGalerkinTimeStepper
    from .labeling import TimeQuadratureLabel
    from .stepper import TimeStepper

    __all__ += [
        "DIRKTimeStepper",
        "BoundsConstrainedDirichletBC",
        "getForm",
        "RadauIIAIMEXMethod",
        "DIRKIMEXMethod",
        "DIRKNystromTimeStepper",
        "ExplicitNystromTimeStepper",
        "StageDerivativeNystromTimeStepper",
        "ClassicNystrom4Tableau",
        "ClinesBase",
        "ClinesLD",
        "IRKAuxiliaryOperatorPC",
        "NystromAuxiliaryOperatorPC",
        "RanaBase",
        "RanaDU",
        "RanaLD",
        "StageValueTimeStepper",
        "ContinuousPetrovGalerkinTimeStepper",
        "DiscontinuousGalerkinTimeStepper",
        "TimeProjector",
        "TimeQuadratureLabel",
        "TimeStepper",
    ]

except ModuleNotFoundError:
    pass
