import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from src.ml.models.base.matrix_a_linear import MatrixALinear
from src.ml.validation.latent_distance_validation import get_roc_auc_for_euclidean_distance_metric, \
    get_roc_auc_for_average_distance_metric, get_roc_auc_for_angle_distance_with_origin_shift
from src.ml.validation.validation_utils import load_data_points


def validate_model(base_path, direction_matrix_path, z, anomalous_direction_indices):
    full_direction_matrix_path = os.path.join(base_path, direction_matrix_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_directions = int(direction_matrix_path.split(".")[0].split("_")[-1])
    direction_matrix: MatrixALinear = MatrixALinear(input_dim=num_directions, output_dim=100, bias=True)
    direction_matrix.load_state_dict(torch.load(full_direction_matrix_path, map_location=torch.device(device)))
    latent_space_data_points, latent_space_data_labels = load_data_points(
        os.path.join(base_path, 'dataset'))

    roc, auc = get_roc_auc_for_euclidean_distance_metric(latent_space_data_points=latent_space_data_points,
                                                         latent_space_data_labels=latent_space_data_labels,
                                                         direction_matrix=direction_matrix,
                                                         anomalous_direction_indices=anomalous_direction_indices,
                                                         z=np.array(z))

    # imgdata = base64.b64decode(roc)
    # img = Image.open(io.BytesIO(imgdata))
    # img.show()
    print("With euclidearn distance", auc)

    roc, auc = get_roc_auc_for_average_distance_metric(latent_space_data_points=latent_space_data_points,
                                                       latent_space_data_labels=latent_space_data_labels,
                                                       direction_matrix=direction_matrix,
                                                       anomalous_direction_indices=anomalous_direction_indices)

    print("With cosine angle metric", auc)

    roc, auc = get_roc_auc_for_angle_distance_with_origin_shift(latent_space_data_points=latent_space_data_points,
                                                                latent_space_data_labels=latent_space_data_labels,
                                                                direction_matrix=direction_matrix,
                                                                anomalous_direction_indices=anomalous_direction_indices,
                                                                z=np.array(z))

    print("With cosine angle metric and origin shift", auc)


def visualize_dataset(base_path, title='', n_components=2):
    latent_space_data_points, latent_space_data_labels = load_data_points(
        os.path.join(base_path, 'dataset'))

    tsne = (TSNE(random_state=42, n_components=n_components, n_iter=300)
            .fit_transform(np.array(latent_space_data_points)))

    plt.figure(figsize=(15, 15))

    if n_components > 2:
        ax = plt.axes(projection='3d')
        ax.scatter3D(tsne[:, 0], tsne[:, 1], tsne[:, 2], c=latent_space_data_labels, cmap='Spectral', marker='.')
    else:
        plt.scatter(tsne[:, 0], tsne[:, 1], c=latent_space_data_labels, cmap='Spectral', marker='.')

    plt.title(title)
    plt.show()
    plt.clf()


noise_mnist_9_6 = [
    0.058331165462732315,
    -1.279407024383545,
    0.45734962821006775,
    -0.23415358364582062,
    0.6843258738517761,
    -1.1026204824447632,
    -0.6358066201210022,
    -1.2379902601242065,
    -0.1359833925962448,
    -0.009202050976455212,
    0.5785013437271118,
    0.01382291130721569,
    -0.32750603556632996,
    -2.036719799041748,
    -0.424043208360672,
    1.582332730293274,
    -0.44165048003196716,
    -0.04944005608558655,
    -0.7427289485931396,
    0.5239644646644592,
    0.06752118468284607,
    -0.11750906705856323,
    0.8048335909843445,
    -0.6029229164123535,
    -1.4870694875717163,
    0.24632592499256134,
    -0.9904230833053589,
    1.8089872598648071,
    -0.30544233322143555,
    1.8801429271697998,
    -0.3863069415092468,
    0.1324443221092224,
    -0.8146682977676392,
    1.0974195003509521,
    -0.3195071220397949,
    -2.1369144916534424,
    0.2711400091648102,
    2.330554485321045,
    0.11651037633419037,
    -1.158079743385315,
    -1.1400402784347534,
    -0.5774142146110535,
    0.5281487107276917,
    0.7373841404914856,
    -1.0360263586044312,
    -0.01035651657730341,
    -1.3251248598098755,
    0.11817290633916855,
    -1.3601973056793213,
    -0.13649719953536987,
    0.23159413039684296,
    1.9963172674179077,
    -0.12799526751041412,
    0.48293182253837585,
    -0.03919629380106926,
    -0.8363949060440063,
    -0.7905920743942261,
    0.8938114047050476,
    0.7369007468223572,
    0.2427479326725006,
    -0.02358781360089779,
    0.09481047838926315,
    0.8031298518180847,
    -1.6757607460021973,
    1.0977121591567993,
    -1.1500732898712158,
    -1.066695213317871,
    -0.8375452756881714,
    -0.16493859887123108,
    0.49839502573013306,
    -0.04094085469841957,
    -0.22853559255599976,
    1.3324710130691528,
    1.196395993232727,
    1.164383888244629,
    1.5984519720077515,
    -0.6063995957374573,
    1.0826091766357422,
    0.11021619290113449,
    -0.20044395327568054,
    0.25930050015449524,
    0.15137644112110138,
    -0.06710505485534668,
    -1.1571167707443237,
    1.766127347946167,
    -0.4255159795284271,
    -1.5525134801864624,
    -1.1163064241409302,
    0.5607522130012512,
    1.0571659803390503,
    0.0762176364660263,
    0.8060373067855835,
    1.3985644578933716,
    0.269676148891449,
    -0.7568891644477844,
    -0.6969798803329468,
    1.2422780990600586,
    1.8914135694503784,
    1.409294605255127,
    -0.33210045099258423
]

anomalous_directions_mnist_9_6 = [
    (12, 1),
    (15, 1),
    (19, 1),
    (20, 1),
    (22, 1),
    (24, 1)]

noise_hazelnut = [
    0.22246648371219635,
    -0.4323245584964752,
    1.8793283700942993,
    -1.1003237962722778,
    -1.326246738433838,
    0.5284701585769653,
    1.596521258354187,
    -0.7805575728416443,
    -0.12192906439304352,
    -1.038532018661499,
    0.6021364331245422,
    0.34105733036994934,
    0.6742216944694519,
    0.3266221582889557,
    -0.09608912467956543,
    0.845346987247467,
    0.29231688380241394,
    -1.9785999059677124,
    -1.5658615827560425,
    1.2457228899002075,
    0.6815308928489685,
    -2.076214551925659,
    0.8859924077987671,
    1.7437745332717896,
    0.7913280129432678,
    -1.760411024093628,
    -1.3143501281738281,
    1.8069349527359009,
    0.25324028730392456,
    1.2596166133880615,
    -0.8177383542060852,
    0.19704633951187134,
    -2.5366058349609375,
    -0.7017405033111572,
    -0.42875349521636963,
    0.8013492226600647,
    -1.3345621824264526,
    -0.08098802715539932,
    -0.46975594758987427,
    0.17904475331306458,
    -0.041060131043195724,
    0.42997434735298157,
    -0.4519161581993103,
    2.4658358097076416,
    -0.33511584997177124,
    -0.08735121041536331,
    0.025638610124588013,
    -0.11292464286088943,
    1.1535052061080933,
    0.26886048913002014,
    -2.1995444297790527,
    -0.3065999448299408,
    -0.32910066843032837,
    -0.08916344493627548,
    -1.201102614402771,
    1.4490095376968384,
    -0.6307165026664734,
    0.31748440861701965,
    0.7793959379196167,
    -1.6723889112472534,
    -0.7520025372505188,
    1.1054035425186157,
    0.25371649861335754,
    -1.4807759523391724,
    1.8748037815093994,
    -0.8357621431350708,
    -0.30258306860923767,
    0.30111151933670044,
    -0.6931483149528503,
    -0.6848558783531189,
    -0.6966869831085205,
    -0.543626606464386,
    1.2507473230361938,
    0.032962437719106674,
    0.28324663639068604,
    -0.7047475576400757,
    -1.1801385879516602,
    -2.243290424346924,
    -0.9163953065872192,
    -0.005137470085173845,
    -0.9982866644859314,
    -0.19700564444065094,
    -1.1724414825439453,
    0.9525427222251892,
    -0.3733370304107666,
    -2.1547555923461914,
    0.4549720287322998,
    0.42323845624923706,
    1.2874367237091064,
    -0.46740660071372986,
    0.4639132618904114,
    0.7472862005233765,
    -1.1158777475357056,
    0.11272294819355011,
    0.6435829997062683,
    -0.7589151859283447,
    1.144984483718872,
    -0.953579843044281,
    0.032908130437135696,
    -0.9116992354393005
]

anomalous_directions_hazelnut = [
    (5, -1),
    (6, 1),
    (7, -1),
    (9, 1),
    (1, -1),
    (1, 1),
    (1, -1),
    (2, -1)
]

noise_fashionmnist_shirt_sneaker = [
    0.7742704749107361,
    -0.4450802803039551,
    -0.4985916018486023,
    -1.146745204925537,
    -0.7779741883277893,
    -0.648368239402771,
    1.243239164352417,
    0.7958066463470459,
    0.6908875703811646,
    -1.056423306465149,
    -0.7950143814086914,
    -1.9319329261779785,
    0.30999648571014404,
    -1.3252466917037964,
    -0.365364134311676,
    1.0125329494476318,
    0.09393145889043808,
    -1.368599534034729,
    0.5612399578094482,
    -1.791019082069397,
    -1.265537142753601,
    0.28361427783966064,
    0.003528253873810172,
    -0.9014045596122742,
    0.492372989654541,
    0.3260788917541504,
    1.5022541284561157,
    1.2841558456420898,
    -0.8667126297950745,
    -0.0625162199139595,
    0.5768486857414246,
    -1.188489317893982,
    -1.1412644386291504,
    -2.3580241203308105,
    -0.8110867738723755,
    1.9679248332977295,
    1.2404110431671143,
    -1.395247220993042,
    0.7604503035545349,
    -0.8417942523956299,
    1.783169150352478,
    -1.0690102577209473,
    -0.8688456416130066,
    -0.2801838219165802,
    -1.1952192783355713,
    0.7123258709907532,
    1.138491153717041,
    0.4974331259727478,
    2.166985511779785,
    0.35690000653266907,
    1.3917442560195923,
    0.6219792366027832,
    -0.9130173921585083,
    -0.6608531475067139,
    0.8240919709205627,
    0.8855801820755005,
    0.491994708776474,
    1.3787446022033691,
    -0.2723062336444855,
    0.8659003376960754,
    -2.0171196460723877,
    -0.6870006918907166,
    1.0302207469940186,
    0.0037988850381225348,
    -0.2401033490896225,
    -0.7034701108932495,
    1.7346588373184204,
    -1.4419618844985962,
    -0.379875123500824,
    1.0125633478164673,
    0.6091403365135193,
    0.5776625871658325,
    2.3293213844299316,
    0.4970360994338989,
    0.9997438788414001,
    0.6448190212249756,
    -0.5505061149597168,
    0.6871010661125183,
    -1.0527143478393555,
    1.0845390558242798,
    1.3665176630020142,
    0.6139785051345825,
    0.3485560417175293,
    1.5158061981201172,
    0.06491503119468689,
    1.274967908859253,
    0.8502529263496399,
    1.1556074619293213,
    0.6216566562652588,
    1.2212642431259155,
    1.1722519397735596,
    -1.4984034299850464,
    1.7153217792510986,
    -0.042723849415779114,
    1.1698027849197388,
    0.14819347858428955,
    0.9256954193115234,
    0.3242723345756531,
    -0.738338828086853,
    0.4538438022136688
]

anomalous_directions_fashionmnist_shirt_sneaker = [
    (1, -1),
    (2, 1),
    (4, -1),
    (5, -1),
    (8, 1),
    (11, 1),
    (14, 1),
    (16, -1),
    (17, 1),
    (19, 1),
    (22, 1),
    (23, 1)
]

# visualize_dataset('../data/DS15_mvtec_hazelnut', title='Hazelnut', n_components=3)
# validate_model('../data/DS15_mvtec_hazelnut', 'direction_matrices/direction_matrix_steps_2500_bias_k_30.pkl',
#                noise_hazelnut,
#                anomalous_directions_hazelnut)

# visualize_dataset('../data/DS12_mnist_9_6', title='MNIST 9 vs 6')
# validate_model('../data/DS12_mnist_9_6', 'direction_matrices/direction_matrix_steps_1500_bias_k_30.pkl',
#                noise_mnist_9_6,
#                anomalous_directions_mnist_9_6)

# visualize_dataset('../data/DS14_fashionmnist_shirt_sneaker', title='FashionMNIST Shirt vs Sneaker')
validate_model('../data/DS14_fashionmnist_shirt_sneaker',
               'direction_matrices/direction_matrix_steps_1500_bias_k_30.pkl',
               noise_fashionmnist_shirt_sneaker,
               anomalous_directions_fashionmnist_shirt_sneaker)

# g = torch.load('/home/yashar/git/AD-with-GANs/data/DS16_mars_novelty/generator_model.pkl')
# test_generator(128, 100, g, '/home/yashar/git/AD-with-GANs/data/DS16_mars_novelty/generator.pkl',
#                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# plt.show(block=False)
# plt.close()
