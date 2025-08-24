import numpy as np

from crowdsim.global_linear_force import GlobalLinearForce
from crowdsim.particle import Particle


def _step(
    particles: list[Particle],
    glf_map: dict[int, GlobalLinearForce],
    dt: float,
) -> list[Particle]:
    new_particles = []
    num = len(particles)
    shape = (num, 2)
    frc_arr = np.empty(shape)
    acc_arr = np.empty(shape)
    vel_arr = np.empty(shape)
    pos_arr = np.empty(shape)

    for i in range(num):
        pos_arr[i] = particles[i].pos
        vel_arr[i] = particles[i].vel

    for i in range(num):
        # calc force
        tgt = particles[i]
        frc = glf_map[tgt.group_idx].force(tgt.pos)
        for j in range(num):
            if j == i:
                continue
            frc += particles[j].force(tgt.pos)

        acc_arr[i] = frc_arr[i] / tgt.mass
        vel_arr[i] += acc_arr[i] * dt
        pos_arr[i] += vel_arr[i] * dt

        new_particles.append(Particle(group_idx=tgt.group_idx, mass=tgt.mass, pos=pos_arr[i], vel=vel_arr[i]))
    return new_particles


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rg = np.random.Generator(np.random.MT19937(1234))
    _num = 10
    _particles = [Particle(group_idx=0, mass=1.0, pos=rg.random(2), vel=np.zeros(2)) for _ in range(_num)]
    _glf_map = {0: GlobalLinearForce(group_idx=0, vec=np.array([1.0, 0.0]))}

    _new_particles = _step(_particles, _glf_map, dt=0.3)
    for i in range(_num):
        plt.plot(
            [_particles[i].pos[0], _new_particles[i].pos[0]],
            [_particles[i].pos[1], _new_particles[i].pos[1]],
            marker="o",
            ls="-",
        )
    plt.show()
