import matplotlib.pyplot as plt
import numpy as np

from crowdsim.aisle import Aisle
from crowdsim.global_linear_force import GlobalLinearForce
from crowdsim.particle import Particle


def _step(
    particles: list[Particle],
    glf_map: dict[int, GlobalLinearForce],
    aisle: Aisle,
    dt: float,
) -> list[Particle]:
    new_particles = []
    num = len(particles)
    shape = (num, 2)
    acc_arr = np.empty(shape)
    vel_arr = np.empty(shape)
    pos_arr = np.empty(shape)

    for i in range(num):
        pos_arr[i] = particles[i].pos
        vel_arr[i] = particles[i].vel

    for i in range(num):
        # calc force
        tgt = particles[i]
        tgt_pos = tgt.pos[np.newaxis, :]
        frc = glf_map[tgt.group_idx].force(tgt_pos)
        for j in range(num):
            if j == i:
                continue
            frc += particles[j].force(tgt_pos)
        frc += aisle.force(tgt_pos)

        acc_arr[i] = frc / tgt.mass
        vel_arr[i] += acc_arr[i] * dt
        pos_arr[i] += vel_arr[i] * dt

        new_particles.append(
            Particle(
                group_idx=tgt.group_idx,
                mass=tgt.mass,
                fric_coef=tgt.fric_coef,
                pos=pos_arr[i],
                vel=vel_arr[i],
            )
        )
    return new_particles


def main() -> None:
    rg = np.random.Generator(np.random.MT19937(1238))
    num = 10
    delta_t = 0.001
    steps = 400
    particles = [
        Particle(
            group_idx=i_p % 2,
            mass=1.0,
            fric_coef=1,
            pos=rg.random(2),
            vel=np.zeros(2),
        )
        for i_p in range(num)
    ]
    glf_map = {
        0: GlobalLinearForce(group_idx=0, vec=np.array([0.0, -500])),
        1: GlobalLinearForce(group_idx=0, vec=np.array([0.0, 500])),
    }
    aisle = Aisle(center=np.array([0.5, 0.5]), direction=np.array([0.0, 1.0]), width=1.0, amp=30.0)

    particle_log = [particles]
    for _ in range(steps):
        particle_log.append(_step(particle_log[-1], glf_map, aisle, dt=delta_t))

    for i_p in range(num):
        plt.plot(
            [particle_log[i_s][i_p].pos[0] for i_s in range(steps + 1)],
            [particle_log[i_s][i_p].pos[1] for i_s in range(steps + 1)],
            ls="-",
        )
        plt.plot(
            [particle_log[-1][i_p].pos[0]],
            [particle_log[-1][i_p].pos[1]],
            marker="o",
        )
    plt.show()


if __name__ == "__main__":
    main()
