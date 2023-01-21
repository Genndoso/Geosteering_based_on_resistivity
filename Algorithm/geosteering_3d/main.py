from differential_evolution.diff_evolution import DE_algo, plot_results
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Differential evolution geosteering implementation')
    parser.add_argument('alg', choices = ['DE', 'RL'], default = 'DE')
    parser.add_argument('cube', help='The resistivity cube file ')
    parser.add_argument('--azimuth_bounds', help = 'Define azimuth lower and high bound', type=tuple,
                        default=(-20, 180))
    parser.add_argument('--zenith_bounds', help='Define zenith lower and high bound', type=tuple,
                        default=(0, 92))
    parser.add_argument('--length', help='Define length of the one step', type=int, default=5)
    parser.add_argument('--angle_constraint', help='Define dogleg severity for the trajectory', default=1.5)
    parser.add_argument('--init_pos', help='Define init position for the trajectory', type=list,
                        default=[20, 150, 20])
    args = parser.parse_args()

    with open(f'{args.cube}.pickle', 'rb') as f:
        cube_3d = pickle.load(f)

    x = np.linspace(0, cube_3d.shape[0] - 1, cube_3d.shape[0])
    y = np.linspace(0, cube_3d.shape[1] - 1, cube_3d.shape[1])
    z = np.linspace(0, cube_3d.shape[2] - 1, cube_3d.shape[2])
    x, y, z = np.meshgrid(x, y, z)

    if args.alg == 'DE':
        DE_algos = DE_algo(cube_3d)

        OFV, traj = DE_algos.DE_planning(angle_constraint=args.angle_constraint,
                                         bounds=[args.azimuth_bounds, args.zenith_bounds], \
                            length = args.length, init_pos = args.init_pos)

        print('Final sum of objective function', OFV)



if __name__ == '__main__':
    main()