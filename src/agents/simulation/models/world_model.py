"""
It allows three world-model categories:

MechanisticWorldModel
LearnedWorldModel
DigitalTwinWorldModel
 
Sources:
- Ha, D., & Schmidhuber, J. (2018). World Models.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.
- Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W. (2018). Digital Twin in manufacturing: A categorical literature review and classification. IFAC-PapersOnLine, 51(11), 1016–1022. DOI: 10.1016/j.ifacol.2018.08.474.
- Tao, F., Zhang, H., Liu, A., & Nee, A. Y. C. (2019). Digital Twin in Industry: State-of-the-Art. IEEE Transactions on Industrial Informatics, 15(4), 2405–2415. DOI: 10.1109/TII.2018.2873186.

The Simulation Agent consumes synchronized physical-state information and implements:
- sensor processing
- pose estimation
- sensor fusion
- SLAM
- object recognition
"""

from __future__ import annotations

__version__ = "2.3.0"

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ...base.modules.physics_constraints import *
from ...base.modules.biology_constraints import *
from ...base.modules.chemistry_constraints import *
from ..utils.config_loader import *
from ..utils.simulation_errors import *
from ..utils.simulation_helpers import *
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("World Model")
printer = PrettyPrinter()


class WorldModelKind(Enum):
    SIMULATION = "simulation"
    LEARNED_WORLD_MODEL = "learned_world_model"
    DIGITAL_MODEL = "digital_model"
    DIGITAL_SHADOW = "digital_shadow"
    DIGITAL_TWIN = "digital_twin"


class WorldModel:
    def __int__(self, config: Optional[Mapping[str, Any]] = None, slaienv: Optional[Any] = None) -> None:
        self.config = load_global_config()
        self.world_config: Dict[str, Any] = dict(get_config_section("world_model") or {})
        if config:
            self.world_config.update(dict(config))
        self.env = slaienv

        logger.info("")

__all__ = ["WorldModel", "WorldModelKind"]