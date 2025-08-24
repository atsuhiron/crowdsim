from collections.abc import Sequence

from crowdsim.global_linear_force import GlobalLinearForce
from crowdsim.particle import Particle


class ParticleGroup:
    def __init__(self, particles: Sequence[Particle], glf: GlobalLinearForce) -> None:
        self.particles = particles
        self.glf = glf

    def set_particles(self, new_particles: Sequence[Particle]) -> None:
        self.particles = new_particles
