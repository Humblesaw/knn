"""
Aircraft Generator for JSBSim / Gymnasium-JSBSim.

Parametrically generates aircraft variants derived from the Cessna C172P.
Physically consistent scaling: weight, wings, engine, moments of inertia.

Usage:
    import gymnasium as gym
    from aircraft_generator import AircraftGenerator, RandomAircraftWrapper
 
    base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
    gen = AircraftGenerator()
    env = RandomAircraftWrapper(base_env, gen)
    obs, info = env.reset()
"""

import copy
import hashlib
import os
import random
import shutil
import tempfile
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import jsbsim as _jsbsim_mod
import numpy as np

from gymnasium_jsbsim.aircraft import Aircraft


JSBSIM_PKG_ROOT = Path(_jsbsim_mod.__file__).parent
BASE_AIRCRAFT_DIR = JSBSIM_PKG_ROOT / "aircraft" / "c172p"
BASE_XML = BASE_AIRCRAFT_DIR / "c172p.xml"


@dataclass
class AircraftParams:
    """
    All scalable aircraft parameters.
    Relative values in comparison with the base c172p model (1.0 = default).
    """

    wing_area_factor: float = 1.0
    wingspan_factor: float = 1.0
    chord_factor: float = 1.0

    htail_area_factor: float = 1.0
    vtail_area_factor: float = 1.0

    empty_weight_factor: float = 1.0

    engine_power_factor: float = 1.0

    cd0_factor: float = 1.0
    c1_alpha_factor: float = 1.0
    cm_alpha_factor: float = 1.0

    _seed: Optional[int] = field(default=None, repr=False)
    _name: str = field(default="", repr=False)

    def cruise_speed_kts(self) -> float:
        base = C172P_BASE["cruise_speed_kts"]
        return base * (self.engine_power_factor ** 0.5) / (self.cd0_factor ** 0.3)

    def summary(self) -> str:
        return (f"[{self._name}] wing_area_factor={self.wing_area_factor:.2f}x, wingspan_factor={self.wingspan_factor:.2f}x, empty_weight_factor={self.empty_weight_factor:.2f}x, engine_power_factor={self.engine_power_factor:.2f}x, cd0_factor={self.cd0_factor:.2f}x")


# base C172P parameters for reference (used for scaling)
C172P_BASE = {
    "wingarea": 174.0,
    "wingspan": 35.8,
    "chord": 4.9,
    "htailarea": 21.9,
    "vtailarea": 16.5,
    "emptywt": 1500.0,

    "ixx": 948.0,
    "iyy": 1346.0,
    "izz": 1967.0,
    "cruise_speed_kts": 120,

    "cd0": 0.027,
    "cdbeta": 0.17,
    "clde": 0.43,
    "clq": 3.9,
    "cmo": 0.1,
    "cmalpha": -1.8,
    "cmq": -12.4,
    "cmde": -1.122,
}


class AircraftGenerator:
    """
    Generates new aircraft based on the C172P template and saves them to a temporary directory.

    Each generated aircraft has a unique name (gen_XXXXXXXX) and is stored at self.output_dir/gen_XXXXXXXX/gen_XXXXXXXX.xml.

    The JSBSim could load the generated aircraft by specifying the path to the generated XML file.
    """

    PARAM_RANGES = {
        "wing_area_factor": (0.7, 1.4),
        "wingspan_factor": (0.75, 1.3),
        "chord_factor": (0.8, 1.25),
        "htail_area_factor": (0.75, 1.3),
        "vtail_area_factor": (0.75, 1.3),
        "empty_weight_factor": (0.65, 1.5),
        "engine_power_factor": (0.7, 1.4),
        "cd0_factor": (0.8, 1.3),
        "c1_alpha_factor": (0.85, 1.2),
        "cm_alpha_factor": (0.8, 1.2),
    }

    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="jsbsim_aircraft_")
            self.root_dir = Path(self._tmpdir)

        else:
            self._tmpdir = None
            self.root_dir = Path(output_dir)

        self.aircraft_dir = self.root_dir / "aircraft"
        self.aircraft_dir.mkdir(parents=True, exist_ok=True)
        self._setup_shared_dir("engine")
        self._setup_shared_dir("systems")
        self._base_tree = ET.parse(BASE_XML)

    def __del__(self):
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def generate(self, params: Optional[AircraftParams] = None, seed: Optional[int] = None) -> Tuple[Aircraft, AircraftParams]:
        """
        Generates a new aircraft by sampling parameters and modifying the base model.

        Args:
            params (Optional[AircraftParams], optional): The explicit parameters to use for generation. Defaults to None.
            seed (Optional[int], optional): The random seed for parameter sampling. Defaults to None.

        Returns:
            (aircraft, params) (Tuple[Aircraft, AircraftParams]): The generated aircraft instance and the parameters used for generation.
        """

        if params is None:
            params = self._random_params(seed)

        name = self._make_name(params)
        params._name = name
        xml_path = self.aircraft_dir / name / f"{name}.xml"

        if not xml_path.exists():
            (self.aircraft_dir / name).mkdir(parents=True, exist_ok=True)
            self._write_xml(params, name, xml_path)
            self._copy_support_files(name)

        aircraft = Aircraft(
            jsbsim_id=name,
            flightgear_id="c172p",
            name=f"generated_{name}",
            cruise_speed_kts=int(params.cruise_speed_kts()),
        )

        return aircraft, params

    def generate_dataset(self, n: int, seed: int = 0) -> list:
        rng = random.Random(seed)
        return [self.generate(seed=rng.randint(0, 2**31)) for _ in range(n)]

    def _random_params(self, seed: Optional[int] = None) -> AircraftParams:
        rng = random.Random(seed)
        p = AircraftParams(_seed=seed)
        for attr, (low, high) in self.PARAM_RANGES.items():
            setattr(p, attr, rng.uniform(low, high))

        return p

    def _make_name(self, params: AircraftParams) -> str:
        key = "_".join(f"{getattr(params, k):.6f}" for k in self.PARAM_RANGES)
        return "gen_" + hashlib.md5(key.encode()).hexdigest()[:8]

    def _setup_shared_dir(self, name: str):
        src = JSBSIM_PKG_ROOT / name
        dst = self.root_dir / name
        if dst.exists() or dst.is_symlink():
            return

        if not src.exists():
            return

        try:
            dst.symlink_to(src)

        except (OSError, NotImplementedError):
            shutil.copytree(src, dst)

    def _copy_support_files(self, name: str):
        dst_dir = self.aircraft_dir / name
        for src_file in BASE_AIRCRAFT_DIR.iterdir():
            if src_file.name == "c172p.xml":
                continue

            dst_file = dst_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)

    def _write_xml(self, p: AircraftParams, name: str, path: Path):
        """
        Modifies the base XML tree according to the provided parameters and writes it to the specified path.
        """

        tree = copy.deepcopy(self._base_tree)
        root = tree.getroot()
        root.set("name", name)

        self._set(root, "metrics/wingarea", C172P_BASE["wingarea"] * p.wing_area_factor)
        self._set(root, "metrics/wingspan", C172P_BASE["wingspan"] * p.wingspan_factor)
        self._set(root, "metrics/chord", C172P_BASE["chord"] * p.chord_factor)
        self._set(root, "metrics/htailarea", C172P_BASE["htailarea"] * p.htail_area_factor)
        self._set(root, "metrics/vtailarea", C172P_BASE["vtailarea"] * p.vtail_area_factor)

        self._set(root, "mass_balance/emptywt", C172P_BASE["emptywt"] * p.empty_weight_factor)

        mf = p.empty_weight_factor
        bf = p.wingspan_factor
        cf = p.chord_factor
        self._set(root, "mass_balance/ixx", C172P_BASE["ixx"] * mf * bf**2)
        self._set(root, "mass_balance/iyy", C172P_BASE["iyy"] * mf * cf**2)
        self._set(root, "mass_balance/izz", C172P_BASE["izz"] * mf * (bf**2 + cf**2) / 2)

        thruster = root.find("propulsion/engine/thruster")
        if thruster is not None:
            pf_el = thruster.find("p_factor")
            if pf_el is not None:
                base_pf = float(pf_el.text.strip())
                pf_el.text = f" {base_pf * p.engine_power_factor:.4f} "

        self._scale_aero(root, "aero/coefficient/CDo", p.cd0_factor)
        self._scale_aero(root, "aero/coefficient/CDbeta", p.cd0_factor)
        self._scale_aero(root, "aero/coefficient/CLDe", p.c1_alpha_factor)
        self._scale_aero(root, "aero/coefficient/CLq", p.c1_alpha_factor)
        self._scale_aero(root, "aero/coefficient/Cmalpha", p.cm_alpha_factor)
        self._scale_aero(root, "aero/coefficient/Cmq", p.cm_alpha_factor)
        self._scale_aero(root, "aero/coefficient/Cmde", p.cm_alpha_factor)

        tree.write(str(path), xml_declaration=True, encoding="utf-8")
        print(f"[AircraftGenerator] generated {name}.xml, {p.summary()}")

    @staticmethod
    def _set(root: ET.Element, xpath: str, value: float):
        """
        Finds the XML element at the specified XPath and scales its text value to the provided value.
        """

        el = root.find(xpath)
        if el is not None:
            el.text = f" {value:.4f} "

    @staticmethod
    def _scale_aero(root: ET.Element, function: str, factor: float):
        """
        Finds <function name="..."> elements in the aerodynamics section and scales the last <value> element by the provided factor.
        """

        for f in root.iter("function"):
            if f.get("name") == function:
                values = list(f.iter("value"))
                if values:
                    last = values[-1]
                    try:
                        v = float(last.text.strip())
                        last.text = f" {v * factor:.6f} "

                    except (ValueError, AttributeError):
                        pass

                break


class RandomAircraftWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that injects a randomly generated aircraft into the JSBSim environment on reset.
    """

    def __init__(self, env: gym.Env, generator: AircraftGenerator, randomize_every_reset: bool = True, seed: Optional[int] = None):
        super().__init__(env)
        self.generator = generator
        self.randomize_every_reset = randomize_every_reset
        self._seed = seed

        self.current_aircraft: Optional[Aircraft] = None
        self.current_params: Optional[AircraftParams] = None
        self._episode_count: int = 0

        # initialization of the first random aircraft
        first_aircraft, first_params = self.generator.generate(seed=seed)
        self.current_aircraft = first_aircraft
        self.current_params = first_params
        self._patch_inner_env(first_aircraft)

    def reset(self, *, seed=None, options=None):
        should_change = self.randomize_every_reset or self._episode_count == 0
        if should_change:
            aircraft, params = self.generator.generate(seed=self._seed)
            self.current_aircraft = aircraft
            self.current_params = params

            # ensure previous JSBSim instance is fully closed
            if hasattr(self.env.unwrapped, "sim") and self.env.unwrapped.sim is not None:
                self.env.unwrapped.sim.close()
                self.env.unwrapped.sim = None

            self._patch_inner_env(aircraft)
            print(f"[RandomAircraftWrapper] episode {self._episode_count:>4d} | {params.summary()}")

        self._episode_count += 1
        return self.env.reset(seed=seed, options=options)

    def _patch_inner_env(self, aircraft: Aircraft):
        jsbsim_env = self.env.unwrapped
        jsbsim_env.aircraft = aircraft
        root_dir = str(self.generator.root_dir)

        env_id = ""
        if hasattr(self.env, "spec") and self.env.spec is not None:
            env_id = self.env.spec.id
        fg_output = "FG" in env_id or "FG" in jsbsim_env.__class__.__name__

        def _patched_init_new_sim(dt, ac, init_conditions):
            from gymnasium_jsbsim.simulation import Simulation
            return Simulation(
                sim_frequency_hz=dt,
                aircraft=aircraft,
                init_conditions=init_conditions,
                allow_flightgear_output=fg_output,
                root_dir=root_dir,
            )

        jsbsim_env._init_new_sim = _patched_init_new_sim


if __name__ == "__main__":
    import gymnasium_jsbsim  # noqa: F401
    
    # 1. Base JSBSim environment
    base_env = gym.make("JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0")
    
    # 2. Add generator logic via our Wrapper
    gen = AircraftGenerator()
    env = RandomAircraftWrapper(base_env, gen, randomize_every_reset=True)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Shape: {obs.shape} | aircraft: {env.current_params.summary()}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1}: reward={reward:.4f}, done={terminated or truncated}")
        if terminated or truncated:
            break

    obs, info = env.reset()
    print(f"Aircraft: {env.current_params.summary()}")
    
    env.close()

    # Possible usage in curriculum.py:
    # ================================
    # from aircraft_generator import AircraftGenerator, RandomAircraftWrapper
    #
    # def make_env(gui: bool = False):
    #     env_name = "JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-FG-v0" if gui else "JSBSim-HeadingControlTask-Cessna172P-Shaping.STANDARD-NoFG-v0"
    #     base_env = gym.make(env_name)
    #
    #     gen = AircraftGenerator()
    #     base_env = RandomAircraftWrapper(base_env, gen)
    #
    #     if STAGE == 1:
    #         return Stage1LevelHeading(base_env)
