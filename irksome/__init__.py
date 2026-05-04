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
from .tableaux.multistep_tableaux import (
    BDF,
    AdamsBashforth,
    AdamsMoulton,
)
from .tableaux.pep_explicit_rk import PEPRK
from .ufl.deriv import Dt, expand_time_derivatives, check_irksome_import_order

check_irksome_import_order()

from .tableaux.dirk_imex_tableaux import DIRK_IMEX
from .tableaux.ars_dirk_imex_tableaux import ARS_DIRK_IMEX
from .tableaux.sspk_tableau import SSPK_DIRK_IMEX, SSPButcherTableau
from .tableaux.multistep_tableaux import MultistepTableau

from .constant import MeshConstant
from .tableaux.wso_dirk_tableaux import WSODIRK
from .scheme import create_time_quadrature
from .scheme import ContinuousPetrovGalerkinScheme, DiscontinuousGalerkinScheme
from .scheme import GalerkinCollocationScheme

from .form_manipulation import getForm

__all__ = [
    "AdamsBashforth",
    "AdamsMoulton",
    "Alexander",
    "ARS_DIRK_IMEX",
    "BackwardEuler",
    "BDF",
    "ContinuousPetrovGalerkinScheme",
    "create_time_quadrature",
    "DIRK_IMEX",
    "DiscontinuousGalerkinScheme",
    "Dt",
    "expand_time_derivatives",
    "getForm",
    "GalerkinCollocationScheme",
    "GaussLegendre",
    "LobattoIIIA",
    "LobattoIIIC",
    "MeshConstant",
    "MultistepTableau",
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
    from .galerkin_stepper import ContinuousPetrovGalerkinTimeStepper
    from .discontinuous_galerkin_stepper import DiscontinuousGalerkinTimeStepper
    from .multistep import MultistepTimeStepper
    from .labeling import TimeQuadratureLabel
    from .stepper import TimeStepper

    __all__ += [
        "DIRKTimeStepper",
        "BoundsConstrainedDirichletBC",
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
        "MultistepTimeStepper",
        "TimeQuadratureLabel",
        "TimeStepper",
    ]

except ModuleNotFoundError:
    pass
