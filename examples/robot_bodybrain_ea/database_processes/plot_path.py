import matplotlib.pyplot as plt
from balance_compute import get_body_states_from_str
from util import open_experiment_table

def plot_core_path(single_robot_body_state_str):
    """Plot the path of the core of a robot."""
    body_states = get_body_states_from_str(single_robot_body_state_str)
    print(body_states)
    core_positions = [body_state.core_position for body_state in body_states]
    x = [core_position.x for core_position in core_positions]
    y = [core_position.y for core_position in core_positions]
    # mark the start and end
    plt.plot(x[0], y[0], "go", label="Start")
    plt.plot(x[-1], y[-1], "ro", label="End")
    plt.legend()
    plt.plot(x, y)
    plt.grid()
    plt.ylim(-1,6)
    plt.xlim(-1,6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Core path")
    plt.savefig("plots/paths/core_path.png", bbox_inches="tight")

def plot_core_path_single_generation(body_states, exp, gen):
    for single_robot_body_state_str in body_states:
        single_robot_states = get_body_states_from_str(single_robot_body_state_str)
        core_positions = [body_state.core_position for body_state in single_robot_states]
        x = [core_position.x for core_position in core_positions]
        y = [core_position.y for core_position in core_positions]
        # mark the start and end
        plt.plot(x[0], y[0], "go")
        plt.plot(x[-1], y[-1], "ro")
        plt.plot(x, y)

    # plot objective
    plt.plot([5], 5, "rx")
    # plt.legend()
    plt.grid(True)
    plt.ylim(-1,6)
    plt.xlim(-1,6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Core path for experiment " + str(exp) + " generation " + str(gen))
    plt.savefig(f"plots/paths/core_path_exp_{exp}_gen_{gen}.png", bbox_inches="tight")
    plt.clf()

def main() -> None:
    """Run the program."""
    df = open_experiment_table()
    # column names: experiment_id, generation_index, fitness, symmetry, xy_positions
    for experiment_id in df["experiment_id"].unique():
        for generation_index in df["generation_index"].unique():
            body_states = df[(df["experiment_id"] == experiment_id) & (df["generation_index"] == generation_index)]["xy_positions"]
            print(f"Experiment {experiment_id}, generation {generation_index}, number of individuals: {len(body_states)}")
            plot_core_path_single_generation(body_states, experiment_id, generation_index)

if __name__ == "__main__":
    main()
    # test_list =["""[BodyState(core_position=Vector3([0.     , 0.     , 0.03015]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([ 3.30150657e-01, -1.27357555e-19,  1.36429106e-01]), core_orientation=Quaternion([-7.06663347e-01,  1.38128454e-17, -7.07549938e-01,8.96811718e-18])), BodyState(core_position=Vector3([ 1.99501207e-01, -6.40151349e-18,  2.22016164e-01]), core_orientation=Quaternion([-9.21097149e-01, -5.39943434e-19, -3.89332817e-01,2.42754960e-18])), BodyState(core_position=Vector3([ 3.06132264e-01, -6.80113107e-18,  1.87046196e-01]), core_orientation=Quaternion([-7.43678923e-01,  6.83138099e-19, -6.68536954e-01,-1.36497578e-18])), BodyState(core_position=Vector3([ 4.45495639e-01, -6.07803134e-18,  3.99182163e-02]), core_orientation=Quaternion([-1.24318487e-01,  4.91125669e-18, -9.92242366e-01,4.03068045e-19])), BodyState(core_position=Vector3([ 4.54827179e-01, -6.13058602e-18,  2.99777896e-02]), core_orientation=Quaternion([-1.40245457e-03,  4.80947085e-18, -9.99999017e-01,4.30060300e-20])), BodyState(core_position=Vector3([ 4.30610800e-01, -6.33329079e-18,  6.97824319e-02]), core_orientation=Quaternion([-2.69306310e-01,  3.92762200e-18, -9.63054573e-01,-2.44674381e-18])), BodyState(core_position=Vector3([ 4.53496104e-01, -6.66434931e-18,  2.87774512e-02]), core_orientation=Quaternion([-1.21381343e-02,  3.17540329e-18, -9.99926330e-01,9.78150482e-19])), BodyState(core_position=Vector3([ 4.47999455e-01, -6.63113412e-18,  4.30040147e-02]), core_orientation=Quaternion([-2.59899206e-02,  3.35939435e-18, -9.99662205e-01,-8.16079851e-20])), BodyState(core_position=Vector3([ 4.49266781e-01, -6.71905001e-18,  3.26151651e-02]), core_orientation=Quaternion([ 3.01340276e-02,  2.91247162e-18, -9.99545867e-01,2.40255787e-19])), BodyState(core_position=Vector3([ 4.47495590e-01, -6.74462799e-18,  3.45828054e-02]), core_orientation=Quaternion([ 5.44186612e-02,  2.92867466e-18, -9.98518207e-01,-2.42412789e-20])), BodyState(core_position=Vector3([ 5.97557141e-01, -7.12072465e-18,  3.53129643e-01]), core_orientation=Quaternion([ 7.06240790e-01, -1.12484168e-18, -7.07971713e-01,2.84177613e-18])), BodyState(core_position=Vector3([ 7.76295582e-01, -1.08839518e-18,  2.05731133e-01]), core_orientation=Quaternion([ 4.18945168e-01,  2.03775573e-17, -9.08011534e-01,-7.20497409e-18])), BodyState(core_position=Vector3([8.29570467e-01, 1.23990890e-17, 5.34743753e-02]), core_orientation=Quaternion([ 4.34208094e-01,  2.60239454e-17, -9.00812595e-01,9.46431036e-18])), BodyState(core_position=Vector3([8.73921602e-01, 3.24507225e-17, 4.71078840e-02]), core_orientation=Quaternion([ 2.35812617e-01,  5.15792376e-17, -9.71798544e-01,1.65432916e-17])), BodyState(core_position=Vector3([8.77607847e-01, 3.47268307e-17, 4.67121691e-02]), core_orientation=Quaternion([ 2.26789044e-01,  5.74830670e-17, -9.73943905e-01,
    #          1.33819360e-17])), BodyState(core_position=Vector3([8.75978181e-01, 4.13537430e-17, 1.06588168e-01]), core_orientation=Quaternion([-7.85333930e-02,  5.92464611e-17, -9.96911484e-01,
    #         -7.02126308e-18])), BodyState(core_position=Vector3([8.34441266e-01, 2.64512698e-17, 1.75388672e-01]), core_orientation=Quaternion([-4.59932102e-01,  3.92028879e-17, -8.87954088e-01,
    #         -2.30921457e-17])), BodyState(core_position=Vector3([9.12781327e-01, 3.99244040e-17, 4.62698177e-02]), core_orientation=Quaternion([-1.67206228e-01,  5.58201216e-17, -9.85921943e-01,
    #         -7.25909721e-18])), BodyState(core_position=Vector3([9.24775049e-01, 4.13022593e-17, 3.37788953e-02]), core_orientation=Quaternion([-4.41746899e-02,  5.28932595e-17, -9.99023822e-01,
    #          4.75523925e-18])), BodyState(core_position=Vector3([9.28837191e-01, 4.09850391e-17, 3.17197400e-02]), core_orientation=Quaternion([ 2.04668570e-02,  4.79030198e-17, -9.99790532e-01,
    #          2.32013176e-18])), BodyState(core_position=Vector3([9.33443156e-01, 4.21739672e-17, 3.87551269e-02]), core_orientation=Quaternion([ 1.07308887e-01,  4.02735554e-17, -9.94225730e-01,
    #          8.35161524e-18])), BodyState(core_position=Vector3([9.42551287e-01, 4.54593272e-17, 4.80211278e-02]), core_orientation=Quaternion([ 2.53975587e-01,  1.80023669e-17, -9.67210629e-01,
    #          1.77738229e-17])), BodyState(core_position=Vector3([9.43779713e-01, 4.59041261e-17, 4.92658084e-02]), core_orientation=Quaternion([ 2.83255878e-01,  4.20162054e-18, -9.59044372e-01,
    #          7.74705374e-18])), BodyState(core_position=Vector3([9.49363772e-01, 4.56810057e-17, 5.17190911e-02]), core_orientation=Quaternion([ 3.49283480e-01, -6.16505467e-19, -9.37017103e-01,
    #          1.25209436e-18])), BodyState(core_position=Vector3([9.68175068e-01, 4.55841073e-17, 5.32075744e-02]), core_orientation=Quaternion([ 5.17806562e-01, -2.38967267e-18, -8.55497729e-01,
    #         -1.05003269e-18])), BodyState(core_position=Vector3([9.84512356e-01, 4.56380080e-17, 4.80059729e-02]), core_orientation=Quaternion([ 6.56337164e-01, -2.07330249e-18, -7.54467711e-01,
    #         -9.43640875e-19])), BodyState(core_position=Vector3([9.90098383e-01, 4.56475288e-17, 4.43697026e-02]), core_orientation=Quaternion([ 7.07777600e-01, -7.14069260e-19, -7.06435326e-01,
    #         -5.86512957e-19])), BodyState(core_position=Vector3([9.91497838e-01, 4.56369053e-17, 4.50379635e-02]), core_orientation=Quaternion([ 7.16256722e-01, -5.22477130e-19, -6.97836878e-01,
    #         -4.44073054e-19])), BodyState(core_position=Vector3([9.75666650e-01, 4.50281480e-17, 5.22653788e-02]), core_orientation=Quaternion([ 5.69161000e-01, -4.43933699e-18, -8.22226098e-01,
    #         -6.27841317e-18])), BodyState(core_position=Vector3([9.78305888e-01, 4.49543199e-17, 4.67426949e-02]), core_orientation=Quaternion([ 2.27559568e-01, -1.95143020e-17, -9.73764162e-01,
    #         -6.17988247e-18])), BodyState(core_position=Vector3([9.98739060e-01, 4.59102978e-17, 4.37390395e-02]), core_orientation=Quaternion([-3.79071004e-02, -1.91339248e-17, -9.99281268e-01,
    #         -4.83529261e-20])), BodyState(core_position=Vector3([9.69286572e-01, 4.53143382e-17, 1.03502901e-01]), core_orientation=Quaternion([-4.52451919e-01, -1.86949015e-17, -8.91788798e-01,
    #          8.63451331e-18])), BodyState(core_position=Vector3([9.75975541e-01, 4.50525152e-17, 9.09194400e-02]), core_orientation=Quaternion([-4.60036614e-01, -1.89916867e-17, -8.87899946e-01,
    #          9.63776256e-18])), BodyState(core_position=Vector3([1.00368680e+00, 4.59847626e-17, 5.36293425e-02]), core_orientation=Quaternion([-2.61639455e-01, -1.63191319e-17, -9.65165683e-01,
    #          2.62693867e-18])), BodyState(core_position=Vector3([1.01228321e+00, 4.67171774e-17, 4.05272642e-02]), core_orientation=Quaternion([-1.30589574e-01, -1.00186078e-17, -9.91436515e-01,
    #         -8.75922092e-19])), BodyState(core_position=Vector3([1.01284832e+00, 4.69826822e-17, 3.98675022e-02]), core_orientation=Quaternion([-1.22436713e-01, -8.15259863e-18, -9.92476323e-01,
    #          2.76622986e-20])), BodyState(core_position=Vector3([1.02007624e+00, 4.71464205e-17, 3.83027963e-02]), core_orientation=Quaternion([-9.87894988e-02, -7.70917450e-18, -9.95108353e-01,
    #         -2.22062907e-19])), BodyState(core_position=Vector3([1.02720029e+00, 4.74201969e-17, 3.68846235e-02]), core_orientation=Quaternion([-8.16347863e-02, -7.10372967e-18, -9.96662311e-01,
    #          2.36279099e-19])), BodyState(core_position=Vector3([1.02612828e+00, 4.72500418e-17, 4.02014530e-02]), core_orientation=Quaternion([-1.27894872e-01, -8.71363254e-18, -9.91787730e-01,
    #          7.02038074e-20])), BodyState(core_position=Vector3([1.02980950e+00, 4.71518716e-17, 3.59442011e-02]), core_orientation=Quaternion([-7.05212043e-02, -8.70606467e-18, -9.97510281e-01,
    #         -1.73838226e-18])), BodyState(core_position=Vector3([1.03914016e+00, 4.72177180e-17, 3.80645136e-02]), core_orientation=Quaternion([ 9.74185662e-02, -9.69325317e-18, -9.95243499e-01,
    #         -1.92764904e-18])), BodyState(core_position=Vector3([1.05537043e+00, 4.71442958e-17, 4.91038670e-02]), core_orientation=Quaternion([ 2.78606350e-01, -9.13062085e-18, -9.60405384e-01,
    #         -4.77552372e-18])), BodyState(core_position=Vector3([1.06740137e+00, 4.70152616e-17, 5.26251752e-02]), core_orientation=Quaternion([ 3.83775457e-01, -8.67744390e-18, -9.23426445e-01,
    #         -5.71751516e-18])), BodyState(core_position=Vector3([1.11616142e+00, 4.10868351e-17, 5.01657350e-02]), core_orientation=Quaternion([ 2.99229069e-01, -1.87526563e-17, -9.54181306e-01,
    #         -6.06801961e-18])), BodyState(core_position=Vector3([1.15534164e+00, 1.90998050e-17, 4.26215981e-02]), core_orientation=Quaternion([ 9.02686347e-02, -4.92860412e-17, -9.95917453e-01,
    #         -6.40815207e-18])), BodyState(core_position=Vector3([1.15691488e+00, 1.92284236e-17, 7.56654645e-02]), core_orientation=Quaternion([-1.63112236e-01, -4.92598183e-17, -9.86607520e-01,
    #          1.30330707e-17])), BodyState(core_position=Vector3([1.18522372e+00, 1.35873274e-17, 3.91517261e-02]), core_orientation=Quaternion([-1.24787047e-01, -4.88704552e-17, -9.92183548e-01,
    #          2.80901955e-20])), BodyState(core_position=Vector3([1.19032895e+00, 1.35899430e-17, 3.52548044e-02]), core_orientation=Quaternion([-6.31547470e-02, -4.27733730e-17, -9.98003746e-01,
    #         -9.65832558e-18])), BodyState(core_position=Vector3([1.27986019e+00, 3.89335562e-18, 4.84912572e-02]), core_orientation=Quaternion([ 7.99710850e-01,  1.10070289e-17, -6.00385340e-01,
    #         -5.05820526e-17])), BodyState(core_position=Vector3([1.29129184e+00, 6.32517363e-18, 5.34654704e-02]), core_orientation=Quaternion([ 8.66620033e-01, -7.86749457e-18, -4.98968655e-01,
    #         -2.98410077e-17])), BodyState(core_position=Vector3([1.30809753e+00, 7.31626290e-18, 4.76144343e-02]), core_orientation=Quaternion([ 9.69749397e-01,  3.61240931e-19, -2.44102656e-01,
    #         -6.40684752e-17])), BodyState(core_position=Vector3([1.31383463e+00, 2.43990230e-17, 3.32216047e-02]), core_orientation=Quaternion([ 9.99326802e-01, -2.52676765e-18, -3.66870949e-02,
    #         -1.60540052e-16])), BodyState(core_position=Vector3([1.28834284e+00, 1.20516055e-16, 3.00782466e-02]), core_orientation=Quaternion([ 9.99999700e-01, -2.36553078e-17,  7.74832458e-04,
    #         -2.69173812e-16])), BodyState(core_position=Vector3([1.27755120e+00, 2.19014565e-16, 3.01858138e-02]), core_orientation=Quaternion([ 9.99999918e-01, -2.03333553e-18,  4.04010020e-04,
    #         -3.85973010e-16])), BodyState(core_position=Vector3([1.27554614e+00, 2.86919776e-16, 4.14029061e-02]), core_orientation=Quaternion([ 9.89850411e-01,  7.55751666e-17,  1.42113210e-01,
    #         -5.26445927e-16])), BodyState(core_position=Vector3([1.27949947e+00, 3.17352173e-16, 5.52663353e-02]), core_orientation=Quaternion([ 9.59928016e-01,  1.93664165e-16,  2.80246686e-01,
    #         -6.53293612e-16])), BodyState(core_position=Vector3([1.28236388e+00, 3.17128403e-16, 6.02629990e-02]), core_orientation=Quaternion([ 9.47535351e-01,  2.29426338e-16,  3.19650996e-01,
    #         -6.80666830e-16])), BodyState(core_position=Vector3([1.27698949e+00, 3.25434023e-16, 5.51174097e-02]), core_orientation=Quaternion([ 9.61052433e-01,  1.89292061e-16,  2.76366098e-01,
    #         -6.90191630e-16])), BodyState(core_position=Vector3([1.24761860e+00, 3.66044257e-16, 3.68008013e-02]), core_orientation=Quaternion([ 9.96236889e-01, -6.28315538e-17, -8.66721430e-02,
    #         -7.11681204e-16])), BodyState(core_position=Vector3([1.18938150e+00, 4.50190159e-16, 4.96835201e-02]), core_orientation=Quaternion([ 7.75427609e-01, -4.60459423e-16, -6.31436476e-01,
    #         -5.55702823e-16])), BodyState(core_position=Vector3([1.11985622e+00, 5.48823534e-16, 4.49686120e-02]), core_orientation=Quaternion([ 1.97323871e-01, -7.10300018e-16, -9.80338355e-01,
    #         -1.56527788e-16])), BodyState(core_position=Vector3([1.10282781e+00, 5.79292612e-16, 8.06428709e-02]), core_orientation=Quaternion([-1.54570247e-01, -7.01302276e-16, -9.87981801e-01,
    #          1.02383120e-16])), BodyState(core_position=Vector3([8.70042381e-01, 8.98537507e-16, 2.42568008e-01]), core_orientation=Quaternion([-9.29936688e-01, -3.12199576e-16, -3.67719670e-01,
    #          5.37131223e-16])), BodyState(core_position=Vector3([1.05060348e+00, 7.50683597e-16, 1.48566081e-01]), core_orientation=Quaternion([-5.72101085e-01, -4.78278887e-16, -8.20183119e-01,
    #          3.17978759e-16])), BodyState(core_position=Vector3([1.09695530e+00, 6.99133249e-16, 7.22871957e-02]), core_orientation=Quaternion([-3.03167310e-01, -5.51988850e-16, -9.52937344e-01,
    #          1.57665629e-16])), BodyState(core_position=Vector3([1.09897417e+00, 7.12651288e-16, 4.40603733e-02]), core_orientation=Quaternion([-1.81827318e-01, -5.42587329e-16, -9.83330477e-01,
    #          8.33052792e-17])), BodyState(core_position=Vector3([1.09638844e+00, 7.57728645e-16, 3.85290872e-02]), core_orientation=Quaternion([-1.01733026e-01, -4.70411068e-16, -9.94811737e-01,
    #          4.59714786e-17])), BodyState(core_position=Vector3([1.11630515e+00, 7.04867498e-16, 3.59538018e-02]), core_orientation=Quaternion([-6.85614106e-02, -5.23977776e-16, -9.97646898e-01,
    #          3.80826317e-17])), BodyState(core_position=Vector3([1.12761796e+00, 6.45986742e-16, 3.82726506e-02]), core_orientation=Quaternion([-9.83451855e-02, -5.95915563e-16, -9.95152362e-01,
    #          5.78632054e-17])), BodyState(core_position=Vector3([1.12979153e+00, 6.27005386e-16, 3.76980715e-02]), core_orientation=Quaternion([-8.44639505e-02, -6.27445636e-16, -9.96426536e-01,
    #          5.50893847e-17])), BodyState(core_position=Vector3([1.13195414e+00, 6.20744797e-16, 3.27361837e-02]), core_orientation=Quaternion([-3.21697031e-02, -6.41283933e-16, -9.99482421e-01,
    #          1.90226431e-17])), BodyState(core_position=Vector3([1.13903759e+00, 6.13856223e-16, 3.86495927e-02]), core_orientation=Quaternion([ 1.03568040e-01, -6.12049723e-16, -9.94622371e-01,
    #         -6.69538236e-17])), BodyState(core_position=Vector3([1.15201844e+00, 6.50048465e-16, 5.11252438e-02]), core_orientation=Quaternion([ 3.25534767e-01, -1.00883357e-16, -9.45530071e-01,
    #         -5.49211017e-17])), BodyState(core_position=Vector3([1.16196327e+00, 6.51821863e-16, 5.35248952e-02]), core_orientation=Quaternion([ 4.53494874e-01, -8.66629262e-17, -8.91258885e-01,
    #         -5.94197149e-17])), BodyState(core_position=Vector3([1.17091976e+00, 6.46722640e-16, 5.36107812e-02]), core_orientation=Quaternion([ 4.35389014e-01, -9.50741955e-17, -9.00242416e-01,
    #         -4.89325707e-17])), BodyState(core_position=Vector3([1.20419082e+00, 6.09504483e-16, 4.67390608e-02]), core_orientation=Quaternion([ 2.26173935e-01, -1.46444839e-16, -9.74086932e-01,
    #         -3.59836588e-17])), BodyState(core_position=Vector3([1.22451497e+00, 6.01224948e-16, 6.14255238e-02]), core_orientation=Quaternion([ 1.52614814e-02, -1.53017118e-16, -9.99883537e-01,
    #         -2.23752525e-18])), BodyState(core_position=Vector3([1.19753430e+00, 6.00566894e-16, 1.30479502e-01]), core_orientation=Quaternion([-3.62846996e-01, -1.52928258e-16, -9.31848731e-01,
    #          4.78307908e-17])), BodyState(core_position=Vector3([1.21787036e+00, 5.90699046e-16, 7.66858058e-02]), core_orientation=Quaternion([-3.27976806e-01, -1.65445563e-16, -9.44685776e-01,
    #          5.38901805e-17])), BodyState(core_position=Vector3([1.23488657e+00, 5.87855420e-16, 4.49583214e-02]), core_orientation=Quaternion([-2.06212143e-01, -1.53283179e-16, -9.78507308e-01,
    #          2.23952277e-17])), BodyState(core_position=Vector3([1.24781310e+00, 5.89235407e-16, 3.14872152e-02]), core_orientation=Quaternion([-1.68185400e-02, -9.05322358e-17, -9.99858558e-01,
    #         -9.09878153e-18])), BodyState(core_position=Vector3([1.24830736e+00, 5.90117611e-16, 3.00651517e-02]), core_orientation=Quaternion([ 1.06162707e-03, -7.49768313e-17, -9.99999436e-01,
    #         -2.17754578e-19])), BodyState(core_position=Vector3([1.24944949e+00, 5.89438418e-16, 3.48379005e-02]), core_orientation=Quaternion([ 5.59822742e-02, -7.47911859e-17, -9.98431763e-01,
    #         -8.17683387e-18])), BodyState(core_position=Vector3([1.25531768e+00, 5.83376197e-16, 4.47756067e-02]), core_orientation=Quaternion([ 1.93852641e-01, -1.40657865e-17, -9.81030659e-01,
    #         -1.38781504e-17])), BodyState(core_position=Vector3([1.25708651e+00, 5.80416518e-16, 4.88834915e-02]), core_orientation=Quaternion([ 2.69963363e-01,  2.24470219e-17, -9.62870595e-01,
    #         -1.08660786e-17])), BodyState(core_position=Vector3([1.22012841e+00, 5.58366415e-16, 5.32959612e-02]), core_orientation=Quaternion([ 4.10296550e-01, -6.67869696e-18, -9.11952159e-01,
    #         -1.38809638e-17])), BodyState(core_position=Vector3([1.08731722e+00, 5.60682389e-16, 4.49223863e-02]), core_orientation=Quaternion([ 7.02159489e-01, -6.01916115e-18, -7.12019700e-01,
    #         -8.26983363e-18])), BodyState(core_position=Vector3([9.89594682e-01, 5.63607045e-16, 6.86606529e-02]), core_orientation=Quaternion([ 7.59757222e-01, -2.80082981e-18, -6.50206862e-01,
    #         -4.02409231e-18])), BodyState(core_position=Vector3([9.82522625e-01, 5.64770700e-16, 9.40464669e-02]), core_orientation=Quaternion([ 9.55272828e-01,  7.89390897e-19, -2.95725927e-01,
    #          1.89863283e-18])), BodyState(core_position=Vector3([1.18485703e+00, 5.67928238e-16, 1.56854500e-01]), core_orientation=Quaternion([ 8.16632290e-01,  4.07175774e-18, -5.77158299e-01,
    #          6.22174978e-18]))]"""]
    # test_str = test_list[0].replace("\n", "")
    # # print(test_str)
    # plot_core_path(test_str)
