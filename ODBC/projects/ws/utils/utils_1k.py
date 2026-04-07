import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
try:
    from fastauc import fastauc as roc_auc_score
except:
    print('loading fastauc fail')
    from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import random
from glob import glob
import subprocess
from typing import Callable, Dict, List, Tuple, Any, Optional
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import faiss

EPS = torch.finfo(torch.float16).eps
INF = torch.finfo(torch.float16).max

IMAGENET_LABEL = {'1': ['n01440764', 'tench'], '2': ['n01443537', 'goldfish'], '3': ['n01484850', 'great_white_shark'], '4': ['n01491361', 'tiger_shark'], '5': ['n01494475', 'hammerhead'], '6': ['n01496331', 'electric_ray'], '7': ['n01498041', 'stingray'], '8': ['n01514668', 'cock'], '9': ['n01514859', 'hen'], '10': ['n01518878', 'ostrich'], '11': ['n01530575', 'brambling'], '12': ['n01531178', 'goldfinch'], '13': ['n01532829', 'house_finch'], '14': ['n01534433', 'junco'], '15': ['n01537544', 'indigo_bunting'], '16': ['n01558993', 'robin'], '17': ['n01560419', 'bulbul'], '18': ['n01580077', 'jay_bird'], '19': ['n01582220', 'magpie'], '20': ['n01592084', 'chickadee'], '21': ['n01601694', 'water_ouzel'], '22': ['n01608432', 'kite'], '23': ['n01614925', 'bald_eagle'], '24': ['n01616318', 'vulture'], '25': ['n01622779', 'great_grey_owl'], '26': ['n01629819', 'European_fire_salamander'], '27': ['n01630670', 'common_newt'], '28': ['n01631663', 'eft'], '29': ['n01632458', 'spotted_salamander'], '30': ['n01632777', 'axolotl'], '31': ['n01641577', 'bullfrog'], '32': ['n01644373', 'tree_frog'], '33': ['n01644900', 'tailed_frog'], '34': ['n01664065', 'loggerhead'], '35': ['n01665541', 'leatherback_turtle'], '36': ['n01667114', 'mud_turtle'], '37': ['n01667778', 'terrapin'], '38': ['n01669191', 'box_turtle'], '39': ['n01675722', 'banded_gecko'], '40': ['n01677366', 'common_iguana'], '41': ['n01682714', 'American_chameleon'], '42': ['n01685808', 'whiptail'], '43': ['n01687978', 'agama'], '44': ['n01688243', 'frilled_lizard'], '45': ['n01689811', 'alligator_lizard'], '46': ['n01692333', 'Gila_monster'], '47': ['n01693334', 'green_lizard'], '48': ['n01694178', 'African_chameleon'], '49': ['n01695060', 'Komodo_dragon'], '50': ['n01697457', 'African_crocodile'], '51': ['n01698640', 'American_alligator'], '52': ['n01704323', 'triceratops'], '53': ['n01728572', 'thunder_snake'], '54': ['n01728920', 'ringneck_snake'], '55': ['n01729322', 'hognose_snake'], '56': ['n01729977', 'green_snake'], '57': ['n01734418', 'king_snake'], '58': ['n01735189', 'garter_snake'], '59': ['n01737021', 'water_snake'], '60': ['n01739381', 'vine_snake'], '61': ['n01740131', 'night_snake'], '62': ['n01742172', 'boa_constrictor'], '63': ['n01744401', 'rock_python'], '64': ['n01748264', 'Indian_cobra'], '65': ['n01749939', 'green_mamba'], '66': ['n01751748', 'sea_snake'], '67': ['n01753488', 'horned_viper'], '68': ['n01755581', 'diamondback'], '69': ['n01756291', 'sidewinder'], '70': ['n01768244', 'trilobite'], '71': ['n01770081', 'harvestman'], '72': ['n01770393', 'scorpion'], '73': ['n01773157', 'black_and_gold_garden_spider'], '74': ['n01773549', 'barn_spider'], '75': ['n01773797', 'garden_spider'], '76': ['n01774384', 'black_widow'], '77': ['n01774750', 'tarantula'], '78': ['n01775062', 'wolf_spider'], '79': ['n01776313', 'tick'], '80': ['n01784675', 'centipede'], '81': ['n01795545', 'black_grouse'], '82': ['n01796340', 'ptarmigan'], '83': ['n01797886', 'ruffed_grouse'], '84': ['n01798484', 'prairie_chicken'], '85': ['n01806143', 'peacock'], '86': ['n01806567', 'quail'], '87': ['n01807496', 'partridge'], '88': ['n01817953', 'African_grey'], '89': ['n01818515', 'macaw'], '90': ['n01819313', 'sulphur-crested_cockatoo'], '91': ['n01820546', 'lorikeet_parrot'], '92': ['n01824575', 'coucal_bird'], '93': ['n01828970', 'bee_eater'], '94': ['n01829413', 'hornbill'], '95': ['n01833805', 'hummingbird'], '96': ['n01843065', 'jacamar'], '97': ['n01843383', 'toucan'], '98': ['n01847000', 'drake'], '99': ['n01855032', 'red-breasted_merganser'], '100': ['n01855672', 'goose'], '101': ['n01860187', 'black_swan'], '102': ['n01871265', 'tusker'], '103': ['n01872401', 'echidna'], '104': ['n01873310', 'platypus'], '105': ['n01877812', 'wallaby'], '106': ['n01882714', 'koala'], '107': ['n01883070', 'wombat'], '108': ['n01910747', 'jellyfish'], '109': ['n01914609', 'sea_anemone'], '110': ['n01917289', 'brain_coral'], '111': ['n01924916', 'flatworm'], '112': ['n01930112', 'nematode'], '113': ['n01943899', 'conch'], '114': ['n01944390', 'snail'], '115': ['n01945685', 'land_slug'], '116': ['n01950731', 'sea_slug'], '117': ['n01955084', 'chiton'], '118': ['n01968897', 'chambered_nautilus'], '119': ['n01978287', 'Dungeness_crab'], '120': ['n01978455', 'rock_crab'], '121': ['n01980166', 'fiddler_crab'], '122': ['n01981276', 'king_crab'], '123': ['n01983481', 'American_lobster'], '124': ['n01984695', 'spiny_lobster'], '125': ['n01985128', 'crayfish'], '126': ['n01986214', 'hermit_crab'], '127': ['n01990800', 'isopod'], '128': ['n02002556', 'white_stork'], '129': ['n02002724', 'black_stork'], '130': ['n02006656', 'spoonbill'], '131': ['n02007558', 'flamingo'], '132': ['n02009229', 'little_blue_heron'], '133': ['n02009912', 'American_egret'], '134': ['n02011460', 'bittern'], '135': ['n02012849', 'crane_bird'], '136': ['n02013706', 'limpkin'], '137': ['n02017213', 'European_gallinule'], '138': ['n02018207', 'American_coot'], '139': ['n02018795', 'bustard'], '140': ['n02025239', 'ruddy_turnstone'], '141': ['n02027492', 'red-backed_sandpiper'], '142': ['n02028035', 'redshank'], '143': ['n02033041', 'dowitcher'], '144': ['n02037110', 'oystercatcher'], '145': ['n02051845', 'pelican'], '146': ['n02056570', 'king_penguin'], '147': ['n02058221', 'albatross'], '148': ['n02066245', 'grey_whale'], '149': ['n02071294', 'killer_whale'], '150': ['n02074367', 'dugong'], '151': ['n02077923', 'sea_lion'], '152': ['n02085620', 'Chihuahua'], '153': ['n02085782', 'Japanese_spaniel'], '154': ['n02085936', 'Maltese_dog'], '155': ['n02086079', 'Pekinese'], '156': ['n02086240', 'Shih-Tzu'], '157': ['n02086646', 'Blenheim_spaniel'], '158': ['n02086910', 'papillon'], '159': ['n02087046', 'toy_terrier'], '160': ['n02087394', 'Rhodesian_ridgeback'], '161': ['n02088094', 'Afghan_hound'], '162': ['n02088238', 'basset'], '163': ['n02088364', 'beagle'], '164': ['n02088466', 'bloodhound'], '165': ['n02088632', 'bluetick'], '166': ['n02089078', 'black-and-tan_coonhound'], '167': ['n02089867', 'Walker_hound'], '168': ['n02089973', 'English_foxhound'], '169': ['n02090379', 'redbone'], '170': ['n02090622', 'borzoi'], '171': ['n02090721', 'Irish_wolfhound'], '172': ['n02091032', 'Italian_greyhound'], '173': ['n02091134', 'whippet'], '174': ['n02091244', 'Ibizan_hound'], '175': ['n02091467', 'Norwegian_elkhound'], '176': ['n02091635', 'otterhound'], '177': ['n02091831', 'Saluki'], '178': ['n02092002', 'Scottish_deerhound'], '179': ['n02092339', 'Weimaraner'], '180': ['n02093256', 'Staffordshire_bullterrier'], '181': ['n02093428', 'American_Staffordshire_terrier'], '182': ['n02093647', 'Bedlington_terrier'], '183': ['n02093754', 'Border_terrier'], '184': ['n02093859', 'Kerry_blue_terrier'], '185': ['n02093991', 'Irish_terrier'], '186': ['n02094114', 'Norfolk_terrier'], '187': ['n02094258', 'Norwich_terrier'], '188': ['n02094433', 'Yorkshire_terrier'], '189': ['n02095314', 'wire-haired_fox_terrier'], '190': ['n02095570', 'Lakeland_terrier'], '191': ['n02095889', 'Sealyham_terrier'], '192': ['n02096051', 'Airedale'], '193': ['n02096177', 'cairn'], '194': ['n02096294', 'Australian_terrier'], '195': ['n02096437', 'Dandie_Dinmont'], '196': ['n02096585', 'Boston_bull'], '197': ['n02097047', 'miniature_schnauzer'], '198': ['n02097130', 'giant_schnauzer'], '199': ['n02097209', 'standard_schnauzer'], '200': ['n02097298', 'Scotch_terrier'], '201': ['n02097474', 'Tibetan_terrier'], '202': ['n02097658', 'silky_terrier'], '203': ['n02098105', 'soft-coated_wheaten_terrier'], '204': ['n02098286', 'West_Highland_white_terrier'], '205': ['n02098413', 'Lhasa'], '206': ['n02099267', 'flat-coated_retriever'], '207': ['n02099429', 'curly-coated_retriever'], '208': ['n02099601', 'golden_retriever'], '209': ['n02099712', 'Labrador_retriever'], '210': ['n02099849', 'Chesapeake_Bay_retriever'], '211': ['n02100236', 'German_short-haired_pointer'], '212': ['n02100583', 'vizsla'], '213': ['n02100735', 'English_setter'], '214': ['n02100877', 'Irish_setter'], '215': ['n02101006', 'Gordon_setter'], '216': ['n02101388', 'Brittany_spaniel'], '217': ['n02101556', 'clumber'], '218': ['n02102040', 'English_springer'], '219': ['n02102177', 'Welsh_springer_spaniel'], '220': ['n02102318', 'cocker_spaniel'], '221': ['n02102480', 'Sussex_spaniel'], '222': ['n02102973', 'Irish_water_spaniel'], '223': ['n02104029', 'kuvasz'], '224': ['n02104365', 'schipperke'], '225': ['n02105056', 'groenendael'], '226': ['n02105162', 'malinois'], '227': ['n02105251', 'briard'], '228': ['n02105412', 'kelpie'], '229': ['n02105505', 'komondor'], '230': ['n02105641', 'Old_English_sheepdog'], '231': ['n02105855', 'Shetland_sheepdog'], '232': ['n02106030', 'collie'], '233': ['n02106166', 'Border_collie'], '234': ['n02106382', 'Bouvier_des_Flandres'], '235': ['n02106550', 'Rottweiler'], '236': ['n02106662', 'German_shepherd'], '237': ['n02107142', 'Doberman'], '238': ['n02107312', 'miniature_pinscher'], '239': ['n02107574', 'Greater_Swiss_Mountain_dog'], '240': ['n02107683', 'Bernese_mountain_dog'], '241': ['n02107908', 'Appenzeller'], '242': ['n02108000', 'EntleBucher'], '243': ['n02108089', 'boxer'], '244': ['n02108422', 'bull_mastiff'], '245': ['n02108551', 'Tibetan_mastiff'], '246': ['n02108915', 'French_bulldog'], '247': ['n02109047', 'Great_Dane'], '248': ['n02109525', 'Saint_Bernard'], '249': ['n02109961', 'Eskimo_dog'], '250': ['n02110063', 'malamute'], '251': ['n02110185', 'Siberian_husky'], '252': ['n02110341', 'dalmatian'], '253': ['n02110627', 'affenpinscher'], '254': ['n02110806', 'basenji'], '255': ['n02110958', 'pug'], '256': ['n02111129', 'Leonberg'], '257': ['n02111277', 'Newfoundland'], '258': ['n02111500', 'Great_Pyrenees'], '259': ['n02111889', 'Samoyed'], '260': ['n02112018', 'Pomeranian'], '261': ['n02112137', 'chow'], '262': ['n02112350', 'keeshond'], '263': ['n02112706', 'Brabancon_griffon'], '264': ['n02113023', 'Pembroke'], '265': ['n02113186', 'Cardigan'], '266': ['n02113624', 'toy_poodle'], '267': ['n02113712', 'miniature_poodle'], '268': ['n02113799', 'standard_poodle'], '269': ['n02113978', 'Mexican_hairless'], '270': ['n02114367', 'timber_wolf'], '271': ['n02114548', 'white_wolf'], '272': ['n02114712', 'red_wolf'], '273': ['n02114855', 'coyote'], '274': ['n02115641', 'dingo'], '275': ['n02115913', 'dhole_dog'], '276': ['n02116738', 'African_hunting_dog'], '277': ['n02117135', 'hyena'], '278': ['n02119022', 'red_fox'], '279': ['n02119789', 'kit_fox'], '280': ['n02120079', 'Arctic_fox'], '281': ['n02120505', 'grey_fox'], '282': ['n02123045', 'tabby_cat'], '283': ['n02123159', 'tiger_cat'], '284': ['n02123394', 'Persian_cat'], '285': ['n02123597', 'Siamese_cat'], '286': ['n02124075', 'Egyptian_cat'], '287': ['n02125311', 'cougar'], '288': ['n02127052', 'lynx'], '289': ['n02128385', 'leopard'], '290': ['n02128757', 'snow_leopard'], '291': ['n02128925', 'jaguar'], '292': ['n02129165', 'lion'], '293': ['n02129604', 'tiger'], '294': ['n02130308', 'cheetah'], '295': ['n02132136', 'brown_bear'], '296': ['n02133161', 'American_black_bear'], '297': ['n02134084', 'ice_bear'], '298': ['n02134418', 'sloth_bear'], '299': ['n02137549', 'mongoose'], '300': ['n02138441', 'meerkat'], '301': ['n02165105', 'tiger_beetle'], '302': ['n02165456', 'ladybug'], '303': ['n02167151', 'ground_beetle'], '304': ['n02168699', 'long-horned_beetle'], '305': ['n02169497', 'leaf_beetle'], '306': ['n02172182', 'dung_beetle'], '307': ['n02174001', 'rhinoceros_beetle'], '308': ['n02177972', 'weevil'], '309': ['n02190166', 'fly'], '310': ['n02206856', 'bee'], '311': ['n02219486', 'ant'], '312': ['n02226429', 'grasshopper'], '313': ['n02229544', 'cricket'], '314': ['n02231487', 'walking_stick'], '315': ['n02233338', 'cockroach'], '316': ['n02236044', 'mantis'], '317': ['n02256656', 'cicada'], '318': ['n02259212', 'leafhopper'], '319': ['n02264363', 'lacewing'], '320': ['n02268443', 'dragonfly'], '321': ['n02268853', 'damselfly'], '322': ['n02276258', 'admiral'], '323': ['n02277742', 'ringlet'], '324': ['n02279972', 'monarch'], '325': ['n02280649', 'cabbage_butterfly'], '326': ['n02281406', 'sulphur_butterfly'], '327': ['n02281787', 'lycaenid'], '328': ['n02317335', 'starfish'], '329': ['n02319095', 'sea_urchin'], '330': ['n02321529', 'sea_cucumber'], '331': ['n02325366', 'wood_rabbit'], '332': ['n02326432', 'hare'], '333': ['n02328150', 'Angora_rabbit'], '334': ['n02342885', 'hamster'], '335': ['n02346627', 'porcupine'], '336': ['n02356798', 'fox_squirrel'], '337': ['n02361337', 'marmot'], '338': ['n02363005', 'beaver'], '339': ['n02364673', 'guinea_pig'], '340': ['n02389026', 'sorrel_horse'], '341': ['n02391049', 'zebra'], '342': ['n02395406', 'hog'], '343': ['n02396427', 'wild_boar'], '344': ['n02397096', 'warthog'], '345': ['n02398521', 'hippopotamus'], '346': ['n02403003', 'ox'], '347': ['n02408429', 'water_buffalo'], '348': ['n02410509', 'bison'], '349': ['n02412080', 'ram'], '350': ['n02415577', 'bighorn'], '351': ['n02417914', 'ibex'], '352': ['n02422106', 'hartebeest'], '353': ['n02422699', 'impala'], '354': ['n02423022', 'gazelle'], '355': ['n02437312', 'Arabian_camel'], '356': ['n02437616', 'llama'], '357': ['n02441942', 'weasel'], '358': ['n02442845', 'mink'], '359': ['n02443114', 'polecat'], '360': ['n02443484', 'black-footed_ferret'], '361': ['n02444819', 'otter'], '362': ['n02445715', 'skunk'], '363': ['n02447366', 'badger'], '364': ['n02454379', 'armadillo'], '365': ['n02457408', 'three-toed_sloth'], '366': ['n02480495', 'orangutan'], '367': ['n02480855', 'gorilla'], '368': ['n02481823', 'chimpanzee'], '369': ['n02483362', 'gibbon'], '370': ['n02483708', 'siamang'], '371': ['n02484975', 'guenon'], '372': ['n02486261', 'patas_monkey'], '373': ['n02486410', 'baboon'], '374': ['n02487347', 'macaque'], '375': ['n02488291', 'langur'], '376': ['n02488702', 'colobus'], '377': ['n02489166', 'proboscis_monkey'], '378': ['n02490219', 'marmoset'], '379': ['n02492035', 'capuchin'], '380': ['n02492660', 'howler_monkey'], '381': ['n02493509', 'titi'], '382': ['n02493793', 'spider_monkey'], '383': ['n02494079', 'squirrel_monkey'], '384': ['n02497673', 'ring_tailed_monkey'], '385': ['n02500267', 'indri_monkey'], '386': ['n02504013', 'Indian_elephant'], '387': ['n02504458', 'African_elephant'], '388': ['n02509815', 'lesser_panda'], '389': ['n02510455', 'giant_panda'], '390': ['n02514041', 'barracouta'], '391': ['n02526121', 'eel'], '392': ['n02536864', 'coho'], '393': ['n02606052', 'rock_beauty'], '394': ['n02607072', 'anemone_fish'], '395': ['n02640242', 'sturgeon'], '396': ['n02641379', 'gar'], '397': ['n02643566', 'lionfish'], '398': ['n02655020', 'puffer'], '399': ['n02666196', 'abacus'], '400': ['n02667093', 'abaya'], '401': ['n02669723', 'academic_gown'], '402': ['n02672831', 'accordion'], '403': ['n02676566', 'acoustic_guitar'], '404': ['n02687172', 'aircraft_carrier'], '405': ['n02690373', 'airliner'], '406': ['n02692877', 'airship'], '407': ['n02699494', 'altar'], '408': ['n02701002', 'ambulance'], '409': ['n02704792', 'amphibian'], '410': ['n02708093', 'analog_clock'], '411': ['n02727426', 'apiary'], '412': ['n02730930', 'apron'], '413': ['n02747177', 'ashcan'], '414': ['n02749479', 'assault_rifle'], '415': ['n02769748', 'backpack'], '416': ['n02776631', 'bakery'], '417': ['n02777292', 'balance_beam'], '418': ['n02782093', 'balloon'], '419': ['n02783161', 'ballpoint'], '420': ['n02786058', 'Band_Aid'], '421': ['n02787622', 'banjo'], '422': ['n02788148', 'bannister'], '423': ['n02790996', 'barbell'], '424': ['n02791124', 'barber_chair'], '425': ['n02791270', 'barbershop'], '426': ['n02793495', 'barn'], '427': ['n02794156', 'barometer'], '428': ['n02795169', 'barrel'], '429': ['n02797295', 'barrow'], '430': ['n02799071', 'baseball'], '431': ['n02802426', 'basketball'], '432': ['n02804414', 'bassinet'], '433': ['n02804610', 'bassoon'], '434': ['n02807133', 'bathing_cap'], '435': ['n02808304', 'bath_towel'], '436': ['n02808440', 'bathtub'], '437': ['n02814533', 'beach_wagon'], '438': ['n02814860', 'beacon'], '439': ['n02815834', 'beaker'], '440': ['n02817516', 'bearskin'], '441': ['n02823428', 'beer_bottle'], '442': ['n02823750', 'beer_glass'], '443': ['n02825657', 'bell_cote'], '444': ['n02834397', 'bib'], '445': ['n02835271', 'bicycle-built-for-two'], '446': ['n02837789', 'bikini'], '447': ['n02840245', 'binder'], '448': ['n02841315', 'binoculars'], '449': ['n02843684', 'birdhouse'], '450': ['n02859443', 'boathouse'], '451': ['n02860847', 'bobsled'], '452': ['n02865351', 'bolo_tie'], '453': ['n02869837', 'bonnet'], '454': ['n02870880', 'bookcase'], '455': ['n02871525', 'bookshop'], '456': ['n02877765', 'bottlecap'], '457': ['n02879718', 'bow'], '458': ['n02883205', 'bow_tie'], '459': ['n02892201', 'brass'], '460': ['n02892767', 'brassiere'], '461': ['n02894605', 'breakwater'], '462': ['n02895154', 'breastplate'], '463': ['n02906734', 'broom'], '464': ['n02909870', 'bucket'], '465': ['n02910353', 'buckle'], '466': ['n02916936', 'bulletproof_vest'], '467': ['n02917067', 'bullet_train'], '468': ['n02927161', 'butcher_shop'], '469': ['n02930766', 'cab'], '470': ['n02939185', 'caldron'], '471': ['n02948072', 'candle'], '472': ['n02950826', 'cannon'], '473': ['n02951358', 'canoe'], '474': ['n02951585', 'can_opener'], '475': ['n02963159', 'cardigan'], '476': ['n02965783', 'car_mirror'], '477': ['n02966193', 'carousel'], '478': ['n02966687', "carpenter_kit"], '479': ['n02971356', 'carton'], '480': ['n02974003', 'car_wheel'], '481': ['n02977058', 'cash_machine'], '482': ['n02978881', 'cassette'], '483': ['n02979186', 'cassette_player'], '484': ['n02980441', 'castle'], '485': ['n02981792', 'catamaran'], '486': ['n02988304', 'CD_player'], '487': ['n02992211', 'cello'], '488': ['n02992529', 'cellular_telephone'], '489': ['n02999410', 'chain'], '490': ['n03000134', 'chainlink_fence'], '491': ['n03000247', 'chain_mail'], '492': ['n03000684', 'chain_saw'], '493': ['n03014705', 'chest'], '494': ['n03016953', 'chiffonier'], '495': ['n03017168', 'chime'], '496': ['n03018349', 'china_cabinet'], '497': ['n03026506', 'Christmas_stocking'], '498': ['n03028079', 'church'], '499': ['n03032252', 'cinema'], '500': ['n03041632', 'cleaver'], '501': ['n03042490', 'cliff_dwelling'], '502': ['n03045698', 'cloak'], '503': ['n03047690', 'clog'], '504': ['n03062245', 'cocktail_shaker'], '505': ['n03063599', 'coffee_mug'], '506': ['n03063689', 'coffeepot'], '507': ['n03065424', 'coil'], '508': ['n03075370', 'combination_lock'], '509': ['n03085013', 'computer_keyboard'], '510': ['n03089624', 'confectionery'], '511': ['n03095699', 'container_ship'], '512': ['n03100240', 'convertible'], '513': ['n03109150', 'corkscrew'], '514': ['n03110669', 'cornet'], '515': ['n03124043', 'cowboy_boot'], '516': ['n03124170', 'cowboy_hat'], '517': ['n03125729', 'cradle'], '518': ['n03126707', 'crane_machine'], '519': ['n03127747', 'crash_helmet'], '520': ['n03127925', 'crate'], '521': ['n03131574', 'crib'], '522': ['n03133878', 'Crock_Pot'], '523': ['n03134739', 'croquet_ball'], '524': ['n03141823', 'crutch'], '525': ['n03146219', 'cuirass'], '526': ['n03160309', 'dam'], '527': ['n03179701', 'desk'], '528': ['n03180011', 'desktop_computer'], '529': ['n03187595', 'dial_telephone'], '530': ['n03188531', 'diaper'], '531': ['n03196217', 'digital_clock'], '532': ['n03197337', 'digital_watch'], '533': ['n03201208', 'dining_table'], '534': ['n03207743', 'dishrag'], '535': ['n03207941', 'dishwasher'], '536': ['n03208938', 'disk_brake'], '537': ['n03216828', 'dock'], '538': ['n03218198', 'dogsled'], '539': ['n03220513', 'dome'], '540': ['n03223299', 'doormat'], '541': ['n03240683', 'drilling_platform'], '542': ['n03249569', 'drum'], '543': ['n03250847', 'drumstick'], '544': ['n03255030', 'dumbbell'], '545': ['n03259280', 'Dutch_oven'], '546': ['n03271574', 'electric_fan'], '547': ['n03272010', 'electric_guitar'], '548': ['n03272562', 'electric_locomotive'], '549': ['n03290653', 'entertainment_center'], '550': ['n03291819', 'envelope'], '551': ['n03297495', 'espresso_maker'], '552': ['n03314780', 'face_powder'], '553': ['n03325584', 'feather_boa'], '554': ['n03337140', 'file'], '555': ['n03344393', 'fireboat'], '556': ['n03345487', 'fire_engine'], '557': ['n03347037', 'fire_screen'], '558': ['n03355925', 'flagpole'], '559': ['n03372029', 'flute'], '560': ['n03376595', 'folding_chair'], '561': ['n03379051', 'football_helmet'], '562': ['n03384352', 'forklift'], '563': ['n03388043', 'fountain'], '564': ['n03388183', 'fountain_pen'], '565': ['n03388549', 'four-poster'], '566': ['n03393912', 'freight_car'], '567': ['n03394916', 'French_horn'], '568': ['n03400231', 'frying_pan'], '569': ['n03404251', 'fur_coat'], '570': ['n03417042', 'garbage_truck'], '571': ['n03424325', 'gasmask'], '572': ['n03425413', 'gas_pump'], '573': ['n03443371', 'goblet'], '574': ['n03444034', 'go-kart'], '575': ['n03445777', 'golf_ball'], '576': ['n03445924', 'golfcart'], '577': ['n03447447', 'gondola'], '578': ['n03447721', 'gong'], '579': ['n03450230', 'gown'], '580': ['n03452741', 'grand_piano'], '581': ['n03457902', 'greenhouse'], '582': ['n03459775', 'grille'], '583': ['n03461385', 'grocery_store'], '584': ['n03467068', 'guillotine'], '585': ['n03476684', 'hair_slide'], '586': ['n03476991', 'hair_spray'], '587': ['n03478589', 'half_track'], '588': ['n03481172', 'hammer'], '589': ['n03482405', 'hamper'], '590': ['n03483316', 'hand_blower'], '591': ['n03485407', 'hand-held_computer'], '592': ['n03485794', 'handkerchief'], '593': ['n03492542', 'hard_disc'], '594': ['n03494278', 'harmonica'], '595': ['n03495258', 'harp'], '596': ['n03496892', 'harvester'], '597': ['n03498962', 'hatchet'], '598': ['n03527444', 'holster'], '599': ['n03529860', 'home_theater'], '600': ['n03530642', 'honeycomb'], '601': ['n03532672', 'hook'], '602': ['n03534580', 'hoopskirt'], '603': ['n03535780', 'horizontal_bar'], '604': ['n03538406', 'horse_cart'], '605': ['n03544143', 'hourglass'], '606': ['n03584254', 'iPod'], '607': ['n03584829', 'iron'], '608': ['n03590841', "jack-o-lantern"], '609': ['n03594734', 'jean'], '610': ['n03594945', 'jeep'], '611': ['n03595614', 'jersey'], '612': ['n03598930', 'jigsaw_puzzle'], '613': ['n03599486', 'jinrikisha'], '614': ['n03602883', 'joystick'], '615': ['n03617480', 'kimono'], '616': ['n03623198', 'knee_pad'], '617': ['n03627232', 'knot'], '618': ['n03630383', 'lab_coat'], '619': ['n03633091', 'ladle'], '620': ['n03637318', 'lampshade'], '621': ['n03642806', 'laptop'], '622': ['n03649909', 'lawn_mower'], '623': ['n03657121', 'lens_cap'], '624': ['n03658185', 'letter_opener'], '625': ['n03661043', 'library'], '626': ['n03662601', 'lifeboat'], '627': ['n03666591', 'lighter'], '628': ['n03670208', 'limousine'], '629': ['n03673027', 'liner'], '630': ['n03676483', 'lipstick'], '631': ['n03680355', 'Loafer'], '632': ['n03690938', 'lotion'], '633': ['n03691459', 'loudspeaker'], '634': ['n03692522', 'loupe'], '635': ['n03697007', 'lumbermill'], '636': ['n03706229', 'magnetic_compass'], '637': ['n03709823', 'mailbag'], '638': ['n03710193', 'mailbox'], '639': ['n03710637', 'maillot_jersey'], '640': ['n03710721', 'maillot_swimsuit'], '641': ['n03717622', 'manhole_cover'], '642': ['n03720891', 'maraca'], '643': ['n03721384', 'marimba'], '644': ['n03724870', 'mask'], '645': ['n03729826', 'matchstick'], '646': ['n03733131', 'maypole'], '647': ['n03733281', 'maze'], '648': ['n03733805', 'measuring_cup'], '649': ['n03742115', 'medicine_chest'], '650': ['n03743016', 'megalith'], '651': ['n03759954', 'microphone'], '652': ['n03761084', 'microwave'], '653': ['n03763968', 'military_uniform'], '654': ['n03764736', 'milk_can'], '655': ['n03769881', 'minibus'], '656': ['n03770439', 'miniskirt'], '657': ['n03770679', 'minivan'], '658': ['n03773504', 'missile'], '659': ['n03775071', 'mitten'], '660': ['n03775546', 'mixing_bowl'], '661': ['n03776460', 'mobile_home'], '662': ['n03777568', 'Model_T'], '663': ['n03777754', 'modem'], '664': ['n03781244', 'monastery'], '665': ['n03782006', 'monitor'], '666': ['n03785016', 'moped'], '667': ['n03786901', 'mortar'], '668': ['n03787032', 'mortarboard'], '669': ['n03788195', 'mosque'], '670': ['n03788365', 'mosquito_net'], '671': ['n03791053', 'motor_scooter'], '672': ['n03792782', 'mountain_bike'], '673': ['n03792972', 'mountain_tent'], '674': ['n03793489', 'mouse'], '675': ['n03794056', 'mousetrap'], '676': ['n03796401', 'moving_van'], '677': ['n03803284', 'muzzle'], '678': ['n03804744', 'nail'], '679': ['n03814639', 'neck_brace'], '680': ['n03814906', 'necklace'], '681': ['n03825788', 'nipple'], '682': ['n03832673', 'notebook'], '683': ['n03837869', 'obelisk'], '684': ['n03838899', 'oboe'], '685': ['n03840681', 'ocarina'], '686': ['n03841143', 'odometer'], '687': ['n03843555', 'oil_filter'], '688': ['n03854065', 'organ'], '689': ['n03857828', 'oscilloscope'], '690': ['n03866082', 'overskirt'], '691': ['n03868242', 'oxcart'], '692': ['n03868863', 'oxygen_mask'], '693': ['n03871628', 'packet'], '694': ['n03873416', 'paddle'], '695': ['n03874293', 'paddlewheel'], '696': ['n03874599', 'padlock'], '697': ['n03876231', 'paintbrush'], '698': ['n03877472', 'pajama'], '699': ['n03877845', 'palace'], '700': ['n03884397', 'panpipe'], '701': ['n03887697', 'paper_towel'], '702': ['n03888257', 'parachute'], '703': ['n03888605', 'parallel_bars'], '704': ['n03891251', 'park_bench'], '705': ['n03891332', 'parking_meter'], '706': ['n03895866', 'passenger_car'], '707': ['n03899768', 'patio'], '708': ['n03902125', 'pay-phone'], '709': ['n03903868', 'pedestal'], '710': ['n03908618', 'pencil_box'], '711': ['n03908714', 'pencil_sharpener'], '712': ['n03916031', 'perfume'], '713': ['n03920288', 'Petri_dish'], '714': ['n03924679', 'photocopier'], '715': ['n03929660', 'pick'], '716': ['n03929855', 'pickelhaube'], '717': ['n03930313', 'picket_fence'], '718': ['n03930630', 'pickup_truck'], '719': ['n03933933', 'pier'], '720': ['n03935335', 'piggy_bank'], '721': ['n03937543', 'pill_bottle'], '722': ['n03938244', 'pillow'], '723': ['n03942813', 'ping-pong_ball'], '724': ['n03944341', 'pinwheel'], '725': ['n03947888', 'pirate'], '726': ['n03950228', 'pitcher'], '727': ['n03954731', 'plane'], '728': ['n03956157', 'planetarium'], '729': ['n03958227', 'plastic_bag'], '730': ['n03961711', 'plate_rack'], '731': ['n03967562', 'plow'], '732': ['n03970156', 'plunger'], '733': ['n03976467', 'Polaroid_camera'], '734': ['n03976657', 'pole'], '735': ['n03977966', 'police_van'], '736': ['n03980874', 'poncho'], '737': ['n03982430', 'pool_table'], '738': ['n03983396', 'pop_bottle'], '739': ['n03991062', 'pot'], '740': ['n03992509', "potter_wheel"], '741': ['n03995372', 'power_drill'], '742': ['n03998194', 'prayer_rug'], '743': ['n04004767', 'printer'], '744': ['n04005630', 'prison'], '745': ['n04008634', 'missile_projectile'], '746': ['n04009552', 'projector'], '747': ['n04019541', 'puck'], '748': ['n04023962', 'punching_bag'], '749': ['n04026417', 'purse'], '750': ['n04033901', 'quill'], '751': ['n04033995', 'quilt'], '752': ['n04037443', 'racer'], '753': ['n04039381', 'racket'], '754': ['n04040759', 'radiator'], '755': ['n04041544', 'radio'], '756': ['n04044716', 'radio_telescope'], '757': ['n04049303', 'rain_barrel'], '758': ['n04065272', 'recreational_vehicle'], '759': ['n04067472', 'reel'], '760': ['n04069434', 'reflex_camera'], '761': ['n04070727', 'refrigerator'], '762': ['n04074963', 'remote_control'], '763': ['n04081281', 'restaurant'], '764': ['n04086273', 'revolver'], '765': ['n04090263', 'rifle'], '766': ['n04099969', 'rocking_chair'], '767': ['n04111531', 'rotisserie'], '768': ['n04116512', 'rubber_eraser'], '769': ['n04118538', 'rugby_ball'], '770': ['n04118776', 'rule'], '771': ['n04120489', 'running_shoe'], '772': ['n04125021', 'safe'], '773': ['n04127249', 'safety_pin'], '774': ['n04131690', 'saltshaker'], '775': ['n04133789', 'sandal'], '776': ['n04136333', 'sarong'], '777': ['n04141076', 'sax'], '778': ['n04141327', 'scabbard'], '779': ['n04141975', 'scale'], '780': ['n04146614', 'school_bus'], '781': ['n04147183', 'schooner'], '782': ['n04149813', 'scoreboard'], '783': ['n04152593', 'screen'], '784': ['n04153751', 'screw'], '785': ['n04154565', 'screwdriver'], '786': ['n04162706', 'seat_belt'], '787': ['n04179913', 'sewing_machine'], '788': ['n04192698', 'shield'], '789': ['n04200800', 'shoe_shop'], '790': ['n04201297', 'shoji'], '791': ['n04204238', 'shopping_basket'], '792': ['n04204347', 'shopping_cart'], '793': ['n04208210', 'shovel'], '794': ['n04209133', 'shower_cap'], '795': ['n04209239', 'shower_curtain'], '796': ['n04228054', 'ski'], '797': ['n04229816', 'ski_mask'], '798': ['n04235860', 'sleeping_bag'], '799': ['n04238763', 'slide_rule'], '800': ['n04239074', 'sliding_door'], '801': ['n04243546', 'slot'], '802': ['n04251144', 'snorkel'], '803': ['n04252077', 'snowmobile'], '804': ['n04252225', 'snowplow'], '805': ['n04254120', 'soap_dispenser'], '806': ['n04254680', 'soccer_ball'], '807': ['n04254777', 'sock'], '808': ['n04258138', 'solar_dish'], '809': ['n04259630', 'sombrero'], '810': ['n04263257', 'soup_bowl'], '811': ['n04264628', 'space_bar'], '812': ['n04265275', 'space_heater'], '813': ['n04266014', 'space_shuttle'], '814': ['n04270147', 'spatula'], '815': ['n04273569', 'speedboat'], '816': ['n04275548', 'spider_web'], '817': ['n04277352', 'spindle'], '818': ['n04285008', 'sports_car'], '819': ['n04286575', 'spotlight'], '820': ['n04296562', 'stage'], '821': ['n04310018', 'steam_locomotive'], '822': ['n04311004', 'steel_arch_bridge'], '823': ['n04311174', 'steel_drum'], '824': ['n04317175', 'stethoscope'], '825': ['n04325704', 'stole'], '826': ['n04326547', 'stone_wall'], '827': ['n04328186', 'stopwatch'], '828': ['n04330267', 'stove'], '829': ['n04332243', 'strainer'], '830': ['n04335435', 'streetcar'], '831': ['n04336792', 'stretcher'], '832': ['n04344873', 'studio_couch'], '833': ['n04346328', 'stupa'], '834': ['n04347754', 'submarine'], '835': ['n04350905', 'suit'], '836': ['n04355338', 'sundial'], '837': ['n04355933', 'sunglass'], '838': ['n04356056', 'sunglasses'], '839': ['n04357314', 'sunscreen'], '840': ['n04366367', 'suspension_bridge'], '841': ['n04367480', 'swab'], '842': ['n04370456', 'sweatshirt'], '843': ['n04371430', 'swimming_trunks'], '844': ['n04371774', 'swing'], '845': ['n04372370', 'switch'], '846': ['n04376876', 'syringe'], '847': ['n04380533', 'table_lamp'], '848': ['n04389033', 'tank'], '849': ['n04392985', 'tape_player'], '850': ['n04398044', 'teapot'], '851': ['n04399382', 'teddy'], '852': ['n04404412', 'television'], '853': ['n04409515', 'tennis_ball'], '854': ['n04417672', 'thatch'], '855': ['n04418357', 'theater_curtain'], '856': ['n04423845', 'thimble'], '857': ['n04428191', 'thresher'], '858': ['n04429376', 'throne'], '859': ['n04435653', 'tile_roof'], '860': ['n04442312', 'toaster'], '861': ['n04443257', 'tobacco_shop'], '862': ['n04447861', 'toilet_seat'], '863': ['n04456115', 'torch'], '864': ['n04458633', 'totem_pole'], '865': ['n04461696', 'tow_truck'], '866': ['n04462240', 'toyshop'], '867': ['n04465501', 'tractor'], '868': ['n04467665', 'trailer_truck'], '869': ['n04476259', 'tray'], '870': ['n04479046', 'trench_coat'], '871': ['n04482393', 'tricycle'], '872': ['n04483307', 'trimaran'], '873': ['n04485082', 'tripod'], '874': ['n04486054', 'triumphal_arch'], '875': ['n04487081', 'trolleybus'], '876': ['n04487394', 'trombone'], '877': ['n04493381', 'tub'], '878': ['n04501370', 'turnstile'], '879': ['n04505470', 'typewriter_keyboard'], '880': ['n04507155', 'umbrella'], '881': ['n04509417', 'unicycle'], '882': ['n04515003', 'upright_piano'], '883': ['n04517823', 'vacuum'], '884': ['n04522168', 'vase'], '885': ['n04523525', 'vault'], '886': ['n04525038', 'velvet'], '887': ['n04525305', 'vending_machine'], '888': ['n04532106', 'vestment'], '889': ['n04532670', 'viaduct'], '890': ['n04536866', 'violin'], '891': ['n04540053', 'volleyball'], '892': ['n04542943', 'waffle_iron'], '893': ['n04548280', 'wall_clock'], '894': ['n04548362', 'wallet'], '895': ['n04550184', 'wardrobe'], '896': ['n04552348', 'warplane'], '897': ['n04553703', 'washbasin'], '898': ['n04554684', 'washer'], '899': ['n04557648', 'water_bottle'], '900': ['n04560804', 'water_jug'], '901': ['n04562935', 'water_tower'], '902': ['n04579145', 'whiskey_jug'], '903': ['n04579432', 'whistle'], '904': ['n04584207', 'wig'], '905': ['n04589890', 'window_screen'], '906': ['n04590129', 'window_shade'], '907': ['n04591157', 'Windsor_tie'], '908': ['n04591713', 'wine_bottle'], '909': ['n04592741', 'wing'], '910': ['n04596742', 'wok'], '911': ['n04597913', 'wooden_spoon'], '912': ['n04599235', 'wool'], '913': ['n04604644', 'worm_fence'], '914': ['n04606251', 'wreck'], '915': ['n04612504', 'yawl'], '916': ['n04613696', 'yurt'], '917': ['n06359193', 'web_site'], '918': ['n06596364', 'comic_book'], '919': ['n06785654', 'crossword_puzzle'], '920': ['n06794110', 'street_sign'], '921': ['n06874185', 'traffic_light'], '922': ['n07248320', 'book_jacket'], '923': ['n07565083', 'menu'], '924': ['n07579787', 'plate'], '925': ['n07583066', 'guacamole'], '926': ['n07584110', 'consomme'], '927': ['n07590611', 'hot_pot'], '928': ['n07613480', 'trifle'], '929': ['n07614500', 'ice_cream'], '930': ['n07615774', 'ice_lolly'], '931': ['n07684084', 'French_loaf'], '932': ['n07693725', 'bagel'], '933': ['n07695742', 'pretzel'], '934': ['n07697313', 'cheeseburger'], '935': ['n07697537', 'hotdog'], '936': ['n07711569', 'mashed_potato'], '937': ['n07714571', 'head_cabbage'], '938': ['n07714990', 'broccoli'], '939': ['n07715103', 'cauliflower'], '940': ['n07716358', 'zucchini'], '941': ['n07716906', 'spaghetti_squash'], '942': ['n07717410', 'acorn_squash'], '943': ['n07717556', 'butternut_squash'], '944': ['n07718472', 'cucumber'], '945': ['n07718747', 'artichoke'], '946': ['n07720875', 'bell_pepper'], '947': ['n07730033', 'cardoon'], '948': ['n07734744', 'mushroom'], '949': ['n07742313', 'Granny_Smith'], '950': ['n07745940', 'strawberry'], '951': ['n07747607', 'orange'], '952': ['n07749582', 'lemon'], '953': ['n07753113', 'fig'], '954': ['n07753275', 'pineapple'], '955': ['n07753592', 'banana'], '956': ['n07754684', 'jackfruit'], '957': ['n07760859', 'custard_apple'], '958': ['n07768694', 'pomegranate'], '959': ['n07802026', 'hay'], '960': ['n07831146', 'carbonara'], '961': ['n07836838', 'chocolate_sauce'], '962': ['n07860988', 'dough'], '963': ['n07871810', 'meat_loaf'], '964': ['n07873807', 'pizza'], '965': ['n07875152', 'potpie'], '966': ['n07880968', 'burrito'], '967': ['n07892512', 'red_wine'], '968': ['n07920052', 'espresso'], '969': ['n07930864', 'cup'], '970': ['n07932039', 'eggnog'], '971': ['n09193705', 'alp'], '972': ['n09229709', 'bubble'], '973': ['n09246464', 'cliff'], '974': ['n09256479', 'coral_reef'], '975': ['n09288635', 'geyser'], '976': ['n09332890', 'lakeside'], '977': ['n09399592', 'promontory'], '978': ['n09421951', 'sandbar'], '979': ['n09428293', 'seashore'], '980': ['n09468604', 'valley'], '981': ['n09472597', 'volcano'], '982': ['n09835506', 'ballplayer'], '983': ['n10148035', 'groom'], '984': ['n10565667', 'scuba_diver'], '985': ['n11879895', 'rapeseed'], '986': ['n11939491', 'daisy'], '987': ['n12057211', "yellow_lady_slipper"], '988': ['n12144580', 'corn'], '989': ['n12267677', 'acorn'], '990': ['n12620546', 'hip'], '991': ['n12768682', 'buckeye'], '992': ['n12985857', 'coral_fungus'], '993': ['n12998815', 'agaric'], '994': ['n13037406', 'gyromitra'], '995': ['n13040303', 'stinkhorn'], '996': ['n13044778', 'earthstar'], '997': ['n13052670', 'hen-of-the-woods'], '998': ['n13054560', 'bolete'], '999': ['n13133613', 'corn_ear'], '1000': ['n15075141', 'toilet_tissue']}

SEMANTIC_TEMPLATE = ['a clean origami {}.',
                     'a photo of a {}.',
                     'This is a photo of a {}',
                     'There is a {} in the scene',
                     'There is the {} in the scene',
                     'a photo of a {} in the scene',
                     'a photo of a small {}.',
                     'a photo of a medium {}.',
                     'a photo of a large {}.',
                     'This is a photo of a small {}.',
                     'This is a photo of a medium {}.',
                     'This is a photo of a large {}.',
                     'There is a small {} in the scene.',
                     'There is a medium {} in the scene.',
                     'There is a large {} in the scene.']

PART_LABEL = {
    "snorkel": ["mouthpiece", "tube", "glasses", "face"],
    "syringe": ["barrel", "plunger", "needle hub", "needle"],
    "binoculars": ["objective lens", "eyepiece", "focus knob", "bridge"],
    "cannon": ["barrel", "wheels", "carriage", "breech"],
    "crane_machine": ["boom", "counterweight", "cab", "hoist", "mast"],
    "desktop_computer": ["tower case", "power button", "ports panel", "vent"],
    "gas_pump": ["nozzle", "hose", "display", "handle"],
    "muzzle": ["strap", "cover", "head", "snout"],
    "seat_belt": ["webbing", "buckle", "seat", "human"],
    "patio": ["paving", "furniture", "canopy", "plant"],
    "home_theater": ["screen", "speakers", "table", "couch"],
    "totem_pole": ["base", "carved figures", "top figure"],
    "grille": ["body", "light", "license plate", "emblem"],
    "mountain_tent": ["canopy", "entrance", "rainfly", "guylines"],
    "scoreboard": ["display panel", "frame", "supports", "digits"],
    "cocktail_shaker": ["tin", "strainer", "cap", "body"],
    "kimono": ["sleeves", "collar", "obi"],
    "whiskey_jug": ["body", "neck", "handle", "stopper"],
    "knee_pad": ["pad", "straps", "leg"],
    "book_jacket": ["cover", "pages", "title", "spine"],
    "crash_helmet": ["shell", "visor", "chin strap", "padding"],
    "vestment": ["chasuble", "stole", "cuffs"],
    "cloak": ["hood", "body", "hem"],
    "scabbard": ["sheath", "belt loop", "mouth", "handle-grip", "blade-knife"],
    "beer_glass": ["rim", "body", "base"],
    "swab": ["shaft", "tip-head", "handle", "strand"],
    "drilling_platform": ["sea", "tower", "supports-legs", "deck"],
    "pencil_box": ["lid", "body", "hinge", "zipper"],
    "punching_bag": ["bag body", "chain strap", "mount"],
    "pencil_sharpener": ["body", "handle", "blade", "shavings receptacle"],
    "shower_cap": ["cap body", "head", "hair", "elastic band"],
    "trolleybus": ["pantograph", "body", "wheels", "windows"],
    "perfume": ["bottle", "spray nozzle", "cap"],
    "crate": ["slats", "frame"],
    "ballpoint": ["tip", "barrel", "cap"],
    "comic_book": ["cover", "pages", "title", "spine"],
    "wooden_spoon": ["bowl", "handle"],
    "ice_lolly": ["stick", "frozen block", "head", "hand"],
    "carbonara": ["spaghetti", "sauce", "marinara", "bolognese", "plate"],
    "caldron": ["body", "handle", "lid", "legs"],
    "backpack": ["compartment", "pocket", "strap", "handle"],
    "banana": ["peel", "flesh", "stem", "tip"],
    "Band_Aid": ["pad", "adhesive", "backing"],
    "shopping_basket": ["body", "handle", "rim", "base", "mesh"],
    "bath_towel": ["body", "hem", "hanger loop", "border"],
    "beer_bottle": ["neck", "body", "label", "cap"],
    "park_bench": ["seat", "backrest", "armrest", "legs"],
    "binder": ["cover", "spine", "rings", "label"],
    "bottlecap": ["top", "side wall", "liner", "bottle"],
    "French_loaf": ["crust", "scored-slashes", "tip", "slice-face", "wrapper", "basket"],
    "broom": ["handle", "brush-bristles", "ferrule", "shaft", "tip-head", "strand"],
    "bucket": ["body", "handle", "rim", "base"],
    "cleaver": ["blade", "table", "board", "handle"],
    "can_opener": ["handle", "cutting-wheel", "gear", "arm", "can"],
    "candle": ["body", "wick", "drip", "base"],
    "cellular_telephone": ["screen", "panel", "camera", "buttons", "speaker"],
    "hamper": ["basket-body", "lid", "handles", "base"],
    "espresso_maker": ["body", "group-head", "portafilter", "steam-wand", "water-tank"],
    "combination_lock": ["dial", "body", "shackle", "number ring"],
    "mouse": ["buttons", "scroll-wheel", "body", "wire"],
    "table_lamp": ["base", "stem", "lampshade", "bulb socket", "wire"],
    "dishrag": ["dish", "body", "hem", "hanger-loop"],
    "doormat": ["door", "surface", "edge", "backing"],
    "Loafer": ["upper", "sole", "heel", "vamp"],
    "power_drill": ["chuck", "body", "trigger", "handle", "battery pack", "wire"],
    "cup": ["rim", "body", "handle", "base"],
    "plate_rack": ["slots", "frame", "base", "plate"] ,
    "envelope": ["body", "flap", "seal area", "window"],
    "electric_fan": ["blades", "grill", "motor", "base"],
    "frying_pan": ["body", "handle", "base", "rim"],
    "gown": ["bodice", "skirt", "sleeves", "neckline"],
    "hand_blower": ["nozzle", "barrel", "handle", "air-intake", "wire"],
    "hammer": ["head", "claw", "face", "handle"],
    "iron": ["soleplate", "tank", "handle", "control panel"],
    "jean": ["waistband", "legs", "pockets", "fly"],
    "computer_keyboard": ["keycaps", "frame", "spacebar", "function-row", "numpad"],
    "ladle": ["bowl", "handle", "hook"],
    "lampshade": ["shade", "rim", "fitting", "lining"],
    "laptop": ["screen", "keyboard", "touchpad", "hinge", "base"],
    "lemon": ["peel", "pulp", "stem-end"],
    "letter_opener": ["blade", "handle", "point-tip"],
    "lighter": ["body", "nozzle", "wheel igniter", "fuel window"],
    "lipstick": ["cap", "tube", "bullet", "base"],
    "matchstick": ["head", "shaft", "box"],
    "measuring_cup": ["body", "handle", "spout", "markings", "table"],
    "microwave": ["door", "control panel", "turntable", "cavity"],
    "mixing_bowl": ["bowl", "rim", "base"],
    "monitor": ["screen", "bezel", "stand", "port", "wire"],
    "coffee_mug": ["handle", "rim", "body", "base", "plate"],
    "nail": ["head", "shank-body", "tip-point"],
    "necklace": ["chain", "pendant", "clasp", "neck"],
    "orange": ["peel", "segments", "stem-end", "seed"],
    "padlock": ["body", "shackle", "keyway"],
    "paintbrush": ["bristles", "ferrule", "handle"],
    "paper_towel": ["sheet", "perforation", "pack"],
    "pill_bottle": ["cap", "body", "label", "rim"],
    "pillow": ["shell", "filling", "seam", "bed", "head"],
    "pitcher": ["spout", "handle", "body", "base"],
    "plastic_bag": ["body", "handles", "gusset"],
    "plate": ["rim", "center", "underside"],
    "plunger": ["handle", "rubber cup", "shaft"],
    "pop_bottle": ["neck", "cap", "body", "label", "base"],
    "space_heater": ["grill", "control panel", "housing", "base"],
    "printer": ["paper-tray", "output-tray", "control-panel", "cartridge"],
    "remote_control": ["buttons", "directional-pad", "battery compartment", "infrared emitter"],
    "rule": ["edge", "markings", "body"],
    "running_shoe": ["upper", "sole", "tongue", "laces"],
    "safety_pin": ["pin", "clasp", "spring"],
    "saltshaker": ["body", "cap", "base", "holes"],
    "sandal": ["sole", "straps", "buckle", "footbed"],
    "screw": ["head", "shank", "thread", "point"],
    "shovel": ["blade", "shaft", "grip", "collar"],
    "sleeping_bag": ["shell", "zipper", "hood", "insulation"],
    "soap_dispenser": ["pump", "bottle", "nozzle", "base"],
    "sock": ["cuff", "arch", "toe", "heel"],
    "soup_bowl": ["rim", "bowl", "base"],
    "spatula": ["blade", "handle", "neck"],
    "loudspeaker": ["cone", "grille", "cabinet", "mount", "magnet"],
    "strainer": ["mesh", "rim", "handle", "hook"],
    "teddy": ["head", "limbs", "body", "face"],
    "suit": ["jacket", "trousers", "lapel", "buttons"],
    "sunglasses": ["lenses", "frame", "temples", "bridge"],
    "sweatshirt": ["hood", "pocket", "cuffs", "hem"],
    "swimming_trunks": ["waistband", "leg openings", "pocket", "drawstring"],
    "jersey": ["pocket", "sleeves", "collar", "logo-number", "hem"],
    "television": ["screen", "bezel", "stand", "back panel"],
    "teapot": ["body", "spout", "handle", "lid"],
    "racket": ["head", "stringbed", "frame", "handle"],
    "toaster": ["slots", "lever", "body", "tray", "grille", "knobs"],
    "toilet_tissue": ["tube", "paper", "perforation"],
    "ashcan": ["lid", "body", "rim", "foot pedal"],
    "tray": ["flat surface", "rim", "handles"],
    "umbrella": ["canopy", "ribs", "shaft", "handle"],
    "vacuum": ["hose", "housing", "brush-head", "handle"],
    "vase": ["neck", "body", "base", "rim"],
    "wallet": ["compartment", "card-slots", "coin-pocket", "fold"],
    "digital_watch": ["face", "strap", "buttons", "buckle"],
    "water_bottle": ["cap", "neck", "body", "base"],
    "dumbbell": ["handle", "weight-plates", "collar"],
    "scale": ["platform", "display", "buttons", "base"],
    "whistle": ["body", "mouthpiece", "pea", "ring"],
    "wine_bottle": ["neck", "body", "closure", "label"],
    "mitten": ["palm", "thumb", "cuff", "insulation"],
    "wok": ["bowl", "handle", "base"],
 "tench": ["head", "dorsal fin", "tail", "pectoral fins", "gills"],
 "goldfish": ["head", "dorsal fin", "tail", "pectoral fins", "eyes"],
 "great_white_shark": ["snout", "dorsal fin", "tail", "pectoral fins", "gill slits", "teeth"],
 "tiger_shark": ["snout", "dorsal fin", "tail", "pectoral fins", "gill slits", "teeth"],
 "hammerhead": ["hammer head", "dorsal fin", "tail", "pectoral fins", "eyes", "teeth"],
 "electric_ray": ["pectoral disc", "tail", "eyes", "mouth", "pectoral fin"],
 "stingray": ["pectoral disc", "tail barb", "eyes", "mouth", "pectoral fin"],
 "cock": ["comb", "wattle", "beak", "wings", "tail", "legs"],
 "hen": ["comb", "wattle", "beak", "wings", "tail", "legs"],
 "ostrich": ["head", "neck", "wings", "legs"],
 "brambling": ["beak", "head", "wings", "tail"],
 "goldfinch": ["beak", "head", "wings", "tail"],
 "house_finch": ["beak", "head", "wings", "tail"],
 "junco": ["beak", "head", "wings", "tail"],
 "indigo_bunting": ["beak", "head", "wings", "tail"],
 "robin": ["beak", "head", "wings", "tail"],
 "bulbul": ["beak", "head", "crest", "wings", "tail"],
 "jay_bird": ["beak", "head", "crest", "wings", "tail"],
 "crane_bird": ["beak", "head", "crest", "wings", "tail"],
 "magpie": ["beak", "head", "long tail", "wings"],
 "chickadee": ["beak", "head", "wings", "tail"],
 "water_ouzel": ["beak", "head", "wings", "tail"],
 "kite": ["beak", "head", "wings", "talons", "tail"],
 "bald_eagle": ["beak", "head", "wings", "talons", "tail"],
 "vulture": ["beak", "head", "neck", "wings", "legs"],
 "great_grey_owl": ["face disc", "beak", "wings", "talons", "tail"],
 "European_fire_salamander": ["head", "limbs", "tail", "eyes"],
 "common_newt": ["head", "limbs", "tail", "eyes"],
 "eft": ["head", "limbs", "tail", "eyes"],
 "spotted_salamander": ["head", "limbs", "tail", "spots"],
 "axolotl": ["head", "gills", "limbs", "tail"],
 "bullfrog": ["head", "eyes", "hind legs", "mouth"],
 "tree_frog": ["head", "toe pads", "hind legs", "eyes"],
 "tailed_frog": ["head", "tail organ", "hind legs", "eyes"],
 "loggerhead": ["head", "shell", "flippers", "tail", "belly"],
 "leatherback_turtle": ["head", "leathery shell", "flippers", "tail", "belly"],
 "mud_turtle": ["head", "shell", "legs", "tail", "belly"],
 "terrapin": ["head", "shell", "legs", "tail", "belly"],
 "box_turtle": ["head", "domed shell", "legs", "tail", "belly"],
 "banded_gecko": ["head", "legs", "tail", "eyes"],
 "common_iguana": ["head crest", "dorsal crest", "legs", "tail", "dewlap"],
 "American_chameleon": ["head helmet", "prehensile tail", "feet", "eyes"],
 "whiptail": ["head", "legs", "tail"],
 "agama": ["head", "dorsal crest", "legs", "tail"],
 "frilled_lizard": ["head", "frill", "legs", "tail"],
 "alligator_lizard": ["head", "legs", "tail"],
 "Gila_monster": ["head", "legs", "tail"],
 "green_lizard": ["head", "legs", "tail"],
 "African_chameleon": ["head helmet", "prehensile tail", "feet", "eyes"],
 "Komodo_dragon": ["head", "neck", "legs", "tail"],
 "African_crocodile": ["snout", "eyes and nostrils", "legs", "tail"],
 "American_alligator": ["snout", "eyes and nostrils", "legs", "tail"],
 "triceratops": ["head frill", "horns", "beak", "legs"],
 "thunder_snake": ["head", "body", "belly", "tail"],
 "ringneck_snake": ["head", "neck ring", "body", "tail"],
 "hognose_snake": ["upturned snout", "head", "body pattern", "tail"],
 "green_snake": ["head", "slender body", "tail"],
 "king_snake": ["head", "banded pattern", "body", "tail"],
 "garter_snake": ["head", "striped body", "tail"],
 "water_snake": ["head", "body pattern", "tail"],
 "vine_snake": ["head", "very slender body", "eyes", "tail"],
 "night_snake": ["head", "body pattern", "tail"],
 "boa_constrictor": ["head", "muscular body", "tail"],
 "rock_python": ["head", "heavy body", "tail"],
 "Indian_cobra": ["head", "hood", "fangs", "tail"],
 "green_mamba": ["head", "slender body", "fangs", "tail"],
 "sea_snake": ["head", "flattened tail", "vent", "tail tip"],
 "horned_viper": ["head", "horn", "fangs", "tail"],
 "diamondback": ["head", "rattle", "fangs", "tail"],
 "sidewinder": ["head", "rattle", "belly scales", "tail"],
 "trilobite": ["head shield", "thoracic segments", "tail shield", "compound eyes"],
 "harvestman": ["body", "legs", "mouthparts"],
 "scorpion": ["pincers", "body", "tail", "legs", "stinger"],
 "black_and_gold_garden_spider": ["head", "abdomen", "legs", "spinnerets"],
 "barn_spider": ["head", "abdomen", "legs", "spinnerets"],
 "garden_spider": ["head", "abdomen", "legs", "spinnerets"],
 "black_widow": ["head", "abdomen", "legs", "spinnerets"],
 "tarantula": ["head", "abdomen", "thick legs", "pedipalps"],
 "wolf_spider": ["head", "abdomen", "legs", "eye cluster"],
 "tick": ["body", "mouthparts", "legs"],
 "centipede": ["head", "body segments", "many legs", "claws"],
 "black_grouse": ["beak", "head comb", "wings", "tail"],
 "ptarmigan": ["beak", "head", "wings", "tail"],
 "ruffed_grouse": ["beak", "head", "ruff", "wings"],
 "prairie_chicken": ["beak", "air sacs", "wings", "tail"],
 "peacock": ["head", "crest", "display tail", "wings"],
 "quail": ["beak", "head crest", "wings", "tail"],
 "partridge": ["beak", "head", "wings", "tail"],
 "African_grey": ["beak", "head", "wings", "tail"],
 "macaw": ["beak", "head", "wings", "long tail", "feet"],
 "sulphur-crested_cockatoo": ["crest", "beak", "head", "wings", "tail"],
 "lorikeet_parrot": ["beak", "head", "wings", "tail"],
 "coucal_bird": ["beak", "head", "long tail", "wings"],
 "bee_eater": ["long beak", "head", "wings", "tail"],
"hornbill": ["beak", "head", "wings", "tail", "legs"],
"hummingbird": ["beak", "head", "wings", "tail"],
"jacamar": ["beak", "head", "wings", "tail"],
"toucan": ["beak", "head", "wings", "tail"],
"drake": ["beak", "head", "wings", "tail", "feet"],
"red-breasted_merganser": ["beak", "head", "wings", "tail", "feet"],
"goose": ["beak", "head", "wings", "tail", "feet"],
"black_swan": ["bill", "head", "neck", "wings", "feet"],
"tusker": ["head", "tusks", "trunk", "ears", "legs"],
"echidna": ["snout", "spines", "legs", "eyes"],
"platypus": ["bill", "webbed feet", "tail", "legs"],
"wallaby": ["head", "ears", "forepaws", "hind legs", "tail"],
"koala": ["head", "ears", "forepaws", "hind legs", "tail"],
"wombat": ["head", "ears", "forepaws", "hind legs", "tail"],
"jellyfish": ["bell", "tentacles", "mouth"],
"sea_anemone": ["base", "tentacles", "mouth"],
"brain_coral": ["lobes", "ridges", "valleys"],
"flatworm": ["head", "tail", "body"],
"nematode": ["head", "tail", "body"],
"conch": ["shell", "aperture", "foot", "tentacles"],
"snail": ["shell", "aperture", "foot", "tentacles"],
"land_slug": ["head", "mantle", "foot", "tentacles"],
"sea_slug": ["head", "mantle", "gills", "foot"],
"chiton": ["shell plates", "girdle", "foot"],
"chambered_nautilus": ["shell", "tentacles", "hood"],
"Dungeness_crab": ["shell", "claws", "walking legs", "antennae"],
"rock_crab": ["shell", "claws", "walking legs", "antennae"],
"fiddler_crab": ["shell", "major claw", "walking legs", "eye stalks"],
"king_crab": ["shell", "large legs", "claws", "antennae"],
"American_lobster": ["shell", "claws", "tail fan", "walking legs", "antennae"],
"spiny_lobster": ["shell", "antennae", "tail fan", "walking legs"],
"crayfish": ["shell", "claws", "tail fan", "walking legs", "antennae"],
"hermit_crab": ["shell", "claws", "walking legs", "antennae"],
"isopod": ["body segments", "legs", "antennae"],
"white_stork": ["beak", "head", "neck", "wings", "legs"],
"black_stork": ["beak", "head", "neck", "wings", "legs"],
"spoonbill": ["bill", "head", "neck", "wings", "legs"],
"flamingo": ["bill", "head", "neck", "wings", "legs"],
"little_blue_heron": ["beak", "head", "neck", "wings", "legs"],
"American_egret": ["beak", "head", "neck", "wings", "legs"],
"bittern": ["beak", "head", "neck", "wings", "legs"],
"limpkin": ["beak", "head", "neck", "wings", "legs"],
"European_gallinule": ["beak", "head", "wings", "legs", "tail"],
"American_coot": ["bill", "head", "wings", "legs", "feet"],
"bustard": ["beak", "head", "wings", "legs", "tail"],
"ruddy_turnstone": ["beak", "head", "wings", "legs", "tail"],
"red-backed_sandpiper": ["beak", "head", "wings", "legs", "tail"],
"redshank": ["beak", "head", "wings", "legs", "tail"],
"dowitcher": ["beak", "head", "wings", "legs", "tail"],
"oystercatcher": ["beak", "head", "wings", "legs", "tail"],
"pelican": ["beak", "head", "throat pouch", "wings", "feet"],
"king_penguin": ["beak", "head", "flippers", "feet", "tail"],
"albatross": ["beak", "head", "wings", "tail", "legs"],
"grey_whale": ["head", "blowhole", "flippers", "tail flukes", "dorsal ridge"],
"killer_whale": ["head", "dorsal fin", "flippers", "tail flukes"],
"dugong": ["head", "flippers", "tail flukes"],
"sea_lion": ["head", "flippers", "whiskers", "tail"],
"Chihuahua": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Japanese_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Maltese_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Pekinese": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Shih-Tzu": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Blenheim_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"papillon": ["head", "ears", "muzzle", "body", "legs", "tail"],
"toy_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Rhodesian_ridgeback": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Afghan_hound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"basset": ["head", "ears", "muzzle", "body", "legs", "tail"],
"beagle": ["head", "ears", "muzzle", "body", "legs", "tail"],
"bloodhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"bluetick": ["head", "ears", "muzzle", "body", "legs", "tail"],
"black-and-tan_coonhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Walker_hound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"English_foxhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"redbone": ["head", "ears", "muzzle", "body", "legs", "tail"],
"borzoi": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Irish_wolfhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Italian_greyhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"whippet": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Ibizan_hound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Norwegian_elkhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"otterhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Saluki": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Scottish_deerhound": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Weimaraner": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Staffordshire_bullterrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"American_Staffordshire_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Bedlington_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Border_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Kerry_blue_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Irish_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Norfolk_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Norwich_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Yorkshire_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"wire-haired_fox_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Lakeland_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Sealyham_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Airedale": ["head", "ears", "muzzle", "body", "legs", "tail"],
"cairn": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Australian_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Dandie_Dinmont": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Boston_bull": ["head", "ears", "muzzle", "body", "legs", "tail"],
"miniature_schnauzer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"giant_schnauzer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"standard_schnauzer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Scotch_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Tibetan_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"silky_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"soft-coated_wheaten_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"West_Highland_white_terrier": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Lhasa": ["head", "ears", "muzzle", "body", "legs", "tail"],
"flat-coated_retriever": ["head", "ears", "muzzle", "body", "legs", "tail"],
"curly-coated_retriever": ["head", "ears", "muzzle", "body", "legs", "tail"],
"golden_retriever": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Labrador_retriever": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Chesapeake_Bay_retriever": ["head", "ears", "muzzle", "body", "legs", "tail"],
"German_short-haired_pointer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"vizsla": ["head", "ears", "muzzle", "body", "legs", "tail"],
"English_setter": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Irish_setter": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Gordon_setter": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Brittany_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"clumber": ["head", "ears", "muzzle", "body", "legs", "tail"],
"English_springer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Welsh_springer_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"cocker_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Sussex_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Irish_water_spaniel": ["head", "ears", "muzzle", "body", "legs", "tail"],
"kuvasz": ["head", "ears", "muzzle", "body", "legs", "tail"],
"schipperke": ["head", "ears", "muzzle", "body", "legs", "tail"],
"groenendael": ["head", "ears", "muzzle", "body", "legs", "tail"],
"malinois": ["head", "ears", "muzzle", "body", "legs", "tail"],
"briard": ["head", "ears", "muzzle", "body", "legs", "tail"],
"kelpie": ["head", "ears", "muzzle", "body", "legs", "tail"],
"komondor": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Old_English_sheepdog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Shetland_sheepdog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"collie": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Border_collie": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Bouvier_des_Flandres": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Rottweiler": ["head", "ears", "muzzle", "body", "legs", "tail"],
"German_shepherd": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Doberman": ["head", "ears", "muzzle", "body", "legs", "tail"],
"miniature_pinscher": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Greater_Swiss_Mountain_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Bernese_mountain_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Appenzeller": ["head", "ears", "muzzle", "body", "legs", "tail"],
"EntleBucher": ["head", "ears", "muzzle", "body", "legs", "tail"],
"boxer": ["head", "ears", "muzzle", "body", "legs", "tail"],
"bull_mastiff": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Tibetan_mastiff": ["head", "ears", "muzzle", "body", "legs", "tail"],
"French_bulldog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Great_Dane": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Saint_Bernard": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Eskimo_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
"malamute": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Siberian_husky": ["head", "ears", "muzzle", "body", "legs", "tail"],
"dalmatian": ["head", "ears", "muzzle", "body", "legs", "tail"],
"affenpinscher": ["head", "ears", "muzzle", "body", "legs", "tail"],
"basenji": ["head", "ears", "muzzle", "body", "legs", "tail"],
"pug": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Leonberg": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Newfoundland": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Great_Pyrenees": ["head", "ears", "muzzle", "body", "legs", "tail"],
"Samoyed": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Pomeranian": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "chow": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "keeshond": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Brabancon_griffon": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Pembroke": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Cardigan": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "toy_poodle": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "miniature_poodle": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "standard_poodle": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Mexican_hairless": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "timber_wolf": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "white_wolf": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "red_wolf": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "coyote": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "dingo": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "dhole_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "African_hunting_dog": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "hyena": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "red_fox": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "kit_fox": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Arctic_fox": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "grey_fox": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "tabby_cat": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "tiger_cat": ["head", "ears", "muzzle", "body", "legs", "tail"],
 "Persian_cat": ["head", "ears", "eyes", "body", "tail"],
 "Siamese_cat": ["head", "ears", "eyes", "body", "tail"],
 "Egyptian_cat": ["head", "ears", "eyes", "body", "tail"],
 "cougar": ["head", "ears", "eyes", "body", "tail"],
 "lynx": ["head", "ears", "eyes", "body", "tail"],
 "leopard": ["head", "ears", "eyes", "body", "tail"],
 "snow_leopard": ["head", "ears", "eyes", "body", "tail"],
 "jaguar": ["head", "ears", "eyes", "body", "tail"],
 "lion": ["head", "mane", "eyes", "body", "tail"],
 "tiger": ["head", "ears", "eyes", "body", "tail"],
 "cheetah": ["head", "ears", "eyes", "body", "tail"],
 "brown_bear": ["head", "ears", "muzzle", "body", "legs"],
 "American_black_bear": ["head", "ears", "muzzle", "body", "legs"],
 "ice_bear": ["head", "ears", "muzzle", "body", "legs"],
 "sloth_bear": ["head", "ears", "muzzle", "body", "legs"],
 "mongoose": ["head", "ears", "muzzle", "body", "tail"],
 "meerkat": ["head", "ears", "eyes", "body", "tail"],
 "tiger_beetle": ["head", "thorax", "abdomen", "legs", "antennae"],
 "ladybug": ["head", "thorax", "wings", "legs", "antennae"],
 "ground_beetle": ["head", "thorax", "abdomen", "legs", "antennae"],
 "long-horned_beetle": ["head", "thorax", "abdomen", "long antennae", "legs"],
 "leaf_beetle": ["head", "thorax", "wings", "legs", "antennae"],
 "dung_beetle": ["head", "thorax", "abdomen", "legs"],
 "rhinoceros_beetle": ["head horn", "thorax", "abdomen", "legs"],
 "weevil": ["head snout", "thorax", "abdomen", "legs", "antennae"],
 "fly": ["head", "thorax", "abdomen", "wings", "legs"],
 "bee": ["head", "thorax", "abdomen", "wings", "legs"],
 "ant": ["head", "thorax", "abdomen", "antennae", "legs"],
 "grasshopper": ["head", "thorax", "abdomen", "hind legs", "wings"],
 "cricket": ["head", "thorax", "abdomen", "hind legs", "wings"],
 "walking_stick": ["head", "thorax", "abdomen", "legs"],
 "cockroach": ["head", "thorax", "abdomen", "legs", "antennae"],
 "mantis": ["head", "thorax", "abdomen", "raptorial forelegs", "wings"],
 "cicada": ["head", "thorax", "abdomen", "wings", "eyes"],
 "leafhopper": ["head", "thorax", "abdomen", "wings", "legs"],
 "lacewing": ["head", "thorax", "abdomen", "wings", "antennae"],
 "dragonfly": ["head", "thorax", "abdomen", "wings", "legs"],
 "damselfly": ["head", "thorax", "abdomen", "wings", "legs"],
 "admiral": ["head", "thorax", "abdomen", "wings", "antennae"],
 "ringlet": ["head", "thorax", "abdomen", "wings", "antennae"],
 "monarch": ["head", "thorax", "abdomen", "wings", "antennae"],
 "cabbage_butterfly": ["head", "thorax", "abdomen", "wings", "antennae"],
 "sulphur_butterfly": ["head", "thorax", "abdomen", "wings", "antennae"],
 "lycaenid": ["head", "thorax", "abdomen", "wings", "antennae"],
 "starfish": ["central disc", "arms", "tube feet"],
 "sea_urchin": ["test", "spines", "tube feet"],
 "sea_cucumber": ["body", "tentacles", "tube feet"],
 "wood_rabbit": ["head", "ears", "eyes", "body", "tail"],
 "hare": ["head", "ears", "eyes", "body", "tail"],
 "Angora_rabbit": ["head", "ears", "body", "legs", "tail"],
 "hamster": ["head", "ears", "eyes", "body", "tail"],
 "porcupine": ["head", "quills", "body", "legs"],
 "fox_squirrel": ["head", "ears", "body", "legs", "tail"],
 "marmot": ["head", "ears", "body", "legs", "tail"],
 "beaver": ["head", "teeth", "body", "tail", "legs"],
 "guinea_pig": ["head", "ears", "body", "legs", "tail"],
 "sorrel_horse": ["head", "mane", "body", "legs", "tail"],
 "zebra": ["head", "mane", "body", "legs", "tail"],
 "hog": ["head", "snout", "body", "legs", "tail"],
 "wild_boar": ["head", "tusks", "body", "legs", "tail"],
 "warthog": ["head", "tusks", "body", "legs", "tail"],
 "hippopotamus": ["head", "eyes and nostrils", "body", "legs", "tail"],
 "ox": ["head", "horns", "body", "legs", "tail"],
 "water_buffalo": ["head", "horns", "body", "legs", "tail"],
 "bison": ["head", "horns", "body", "legs", "tail"],
 "ram": ["head", "horns", "body", "legs", "tail"],
 "bighorn": ["head", "curved horns", "body", "legs", "tail"],
 "ibex": ["head", "horns", "body", "legs", "tail"],
 "hartebeest": ["head", "horns", "body", "legs", "tail"],
 "impala": ["head", "horns", "body", "legs", "tail"],
 "gazelle": ["head", "horns", "body", "legs", "tail"],
 "Arabian_camel": ["head", "hump", "neck", "legs", "tail"],
 "llama": ["head", "neck", "body", "legs", "tail"],
 "weasel": ["head", "body", "legs", "tail"],
 "mink": ["head", "body", "legs", "tail"],
 "polecat": ["head", "body", "legs", "tail"],
 "black-footed_ferret": ["head", "body", "legs", "tail"],
 "otter": ["head", "body", "legs", "tail"],
 "skunk": ["head", "body", "legs", "tail"],
 "badger": ["head", "body", "legs", "tail"],
 "armadillo": ["head", "armor plates", "legs", "tail"],
 "three-toed_sloth": ["head", "forelimbs", "hindlimbs", "claws"],
 "orangutan": ["head", "torso", "arms", "legs", "face"],
 "gorilla": ["head", "torso", "arms", "legs", "face"],
 "chimpanzee": ["head", "torso", "arms", "legs", "face"],
 "gibbon": ["head", "torso", "arms", "legs", "face"],
 "siamang": ["head", "torso", "arms", "legs", "face"],
 "guenon": ["head", "torso", "arms", "legs", "tail"],
 "patas_monkey": ["head", "torso", "arms", "legs", "tail"],
 "baboon": ["head", "torso", "arms", "legs", "tail"],
 "macaque": ["head", "torso", "arms", "legs", "tail"],
 "langur": ["head", "torso", "arms", "legs", "tail"],
 "colobus": ["head", "torso", "arms", "legs", "tail"],
 "proboscis_monkey": ["head", "nose", "torso", "arms", "legs"],
 "marmoset": ["head", "torso", "arms", "legs", "tail"],
 "capuchin": ["head", "torso", "arms", "legs", "tail"],
 "howler_monkey": ["head", "torso", "arms", "legs", "tail"],
 "titi": ["head", "torso", "arms", "legs", "tail"],
 "spider_monkey": ["head", "torso", "arms", "legs", "tail"],
 "squirrel_monkey": ["head", "torso", "arms", "legs", "tail"],
 "ring_tailed_monkey": ["head", "torso", "arms", "legs", "tail"],
 "indri_monkey": ["head", "torso", "arms", "legs", "tail"],
 "Indian_elephant": ["head", "trunk", "ears", "tusks", "legs"],
 "African_elephant": ["head", "trunk", "ears", "tusks", "legs"],
 "lesser_panda": ["head", "ears", "body", "legs", "tail"],
 "giant_panda": ["head", "ears", "body", "legs", "tail"],
 "barracouta": ["head", "body", "tail", "fins"],
 "eel": ["head", "body", "tail"],
 "coho": ["head", "body", "tail", "fins"],
 "rock_beauty": ["head", "body", "tail", "fins"],
 "anemone_fish": ["head", "body", "tail", "fins"],
 "sturgeon": ["head", "body", "tail", "fins"],
 "gar": ["head", "body", "tail", "fins"],
 "lionfish": ["head", "body", "tail", "fins", "spines"],
 "puffer": ["head", "body", "tail", "fins"],
 "abacus": ["frame", "rods", "beads"],
 "abaya": ["body", "sleeves", "hem", "front opening"],
 "academic_gown": ["yoke", "sleeves", "body", "hem"],
 "accordion": ["bellows", "keyboard", "bass buttons", "grille"],
 "acoustic_guitar": ["body", "neck", "fretboard", "headstock", "strings"],
 "aircraft_carrier": ["flight deck", "island", "hull", "elevator"],
 "airliner": ["nose", "fuselage", "wings", "tail", "engines"],
 "airship": ["envelope", "gondola", "fins", "propellers"],
 "altar": ["table", "top surface", "steps", "ornament"],
 "ambulance": ["cab", "patient compartment", "lights", "wheels"],
 "amphibian": ["head", "body", "limbs"],
 "analog_clock": ["face", "hands", "numbers", "bezel"],
 "apiary": ["hive boxes", "frames", "entrance"],
 "apron": ["body", "neck strap", "ties", "pocket"],
 "assault_rifle": ["stock", "barrel", "magazine", "sights"],
 "bakery": ["display case", "oven", "counter", "shelves"],
 "balance_beam": ["beam", "supports", "base"],
 "balloon": ["envelope", "knot", "string", "valve"],
 "banjo": ["body", "neck", "head", "strings"],
 "bannister": ["handrail", "balusters", "newel post"],
 "barbell": ["bar", "weight plates", "collars"],
 "barber_chair": ["seat", "backrest", "armrests", "footrest", "lever"],
 "barbershop": ["chair", "mirror", "sink", "counter"],
 "barn": ["roof", "walls", "doors", "loft"],
 "barometer": ["dial", "glass cover", "case"],
 "barrel": ["staves", "hoops", "heads"],
 "barrow": ["wheel", "tray", "handles", "legs"],
 "baseball": ["leather cover", "stitches", "core"],
 "basketball": ["surface", "seams"],
 "bassinet": ["frame", "mattress", "hood"],
 "bassoon": ["body", "crook", "bell", "keys"],
 "bathing_cap": ["cap body", "rim"],
 "bathtub": ["bowl", "rim", "drain", "faucet"],
 "beach_wagon": ["bed", "wheels", "handle"],
 "beacon": ["light", "housing", "base"],
 "beaker": ["body", "lip", "base", "graduations"],
 "bearskin": ["fur cap", "chin strap", "rim"],
 "bell_cote": ["opening", "roof", "support"],
 "bib": ["body", "neck strap", "ties"],
 "bicycle-built-for-two": ["frame", "seats", "handlebars", "wheels", "chain"],
 "bikini": ["top", "bottom", "ties"],
 "birdhouse": ["entrance", "roof", "body", "perch"],
 "boathouse": ["slip", "roof", "walls", "door"],
 "bobsled": ["nose", "cockpit", "runners", "shell"],
 "bolo_tie": ["cord", "slide", "tips"],
 "bonnet": ["hood", "rim", "ties"],
 "bookcase": ["shelves", "back", "sides", "top"],
 "bookshop": ["shelves", "counter", "display", "entrance"],
 "bow": ["limb", "string", "grip", "arrow rest"],
 "bow_tie": ["knot", "wings", "band"],
 "brass": ["bell", "mouthpiece", "valves", "tubing"],
 "brassiere": ["cups", "band", "straps", "hook"],
 "breakwater": ["core", "armor blocks", "seaward face"],
 "breastplate": ["chest panel", "shoulder straps", "lower edge"],
 "buckle": ["frame", "prong", "bar"],
 "bulletproof_vest": ["front panel", "back panel", "shoulder straps"],
 "bullet_train": ["nose", "windows", "doors", "pantograph"],
 "butcher_shop": ["counter", "display case", "hooks", "scale"],
 "cab": ["roof", "doors", "windows", "wheel"],
 "canoe": ["bow", "stern", "hull", "thwarts"],
 "cardigan": ["body", "sleeves", "buttons", "collar"],
 "car_mirror": ["glass", "housing", "arm", "mount"],
 "carousel": ["platform", "poles", "animals", "canopy"],
 "carpenter_kit": ["hammer", "saw", "nails", "screwdriver"],
 "carton": ["walls", "top flap", "bottom flap"],
 "car_wheel": ["rim", "tire", "hub", "spokes"],
 "cash_machine": ["screen", "keypad", "cash slot", "card slot"],
 "cassette": ["case", "spool holes", "tape"],
 "cassette_player": ["cassette slot", "buttons", "display", "speaker"],
 "castle": ["tower", "wall", "gate", "keep"],
 "catamaran": ["hulls", "deck", "mast", "crossbeam"],
 "CD_player": ["tray", "buttons", "display"],
 "cello": ["body", "neck", "fingerboard", "strings", "bridge"],
 "chain": ["links", "end links"],
 "chainlink_fence": ["posts", "mesh", "top rail"],
 "chain_mail": ["rings", "coif", "hauberk"],
 "chain_saw": ["bar", "chain", "engine housing", "handle"],
 "chest": ["lid", "body", "handles", "lock"],
 "chiffonier": ["drawers", "top", "sides", "legs"],
 "chime": ["tubes", "frame", "striker"],
 "china_cabinet": ["glass doors", "shelves", "base", "top"],
 "Christmas_stocking": ["body", "cuff", "loop"],
 "church": ["tower", "nave", "roof", "entrance"],
 "cinema": ["screen", "seats", "aisle", "lobby"],
 "cliff_dwelling": ["rooms", "terraces", "walls"],
 "clog": ["upper", "sole", "heel"],
 "coffeepot": ["body", "spout", "handle", "lid"],
 "coil": ["turns", "core", "lead"],
 "confectionery": ["display", "counter", "shelves"],
 "container_ship": ["deck", "hull", "bridge", "containers"],
 "convertible": ["hood", "windshield", "seats", "wheels"],
 "corkscrew": ["spiral", "handle", "lever"],
 "cornet": ["bell", "mouthpiece", "valves", "tubing"],
 "cowboy_boot": ["upper", "heel", "toe", "pull tabs"],
 "cowboy_hat": ["brim", "crown", "band"],
 "cradle": ["frame", "rocker", "slats", "mattress"],
 "crib": ["slats", "mattress", "rail"],
 "Crock_Pot": ["pot", "lid", "handle", "base", "dial"],
 "croquet_ball": ["surface"],
 "crutch": ["handle", "shaft", "pad", "tip"],
 "cuirass": ["chest panel", "shoulder straps", "lower edge"],
 "dam": ["crest", "face", "spillway"],
 "desk": ["top", "drawers", "legs", "keyboard tray"],
 "dial_telephone": ["handset", "dial", "base", "cord"],
 "diaper": ["absorbent pad", "waistband", "fasteners"],
 "digital_clock": ["display", "buttons", "case"],
 "dining_table": ["top", "legs", "apron"],
 "dishwasher": ["door", "rack", "control panel", "spray arm"],
 "disk_brake": ["rotor", "caliper", "pad", "hub"],
 "dock": ["deck", "pilings", "cleats"],
 "dogsled": ["sled", "runners", "harness attachments"],
 "dome": ["shell", "drum", "opening"],
 "drum": ["shell", "head", "rim", "lugs"],
 "drumstick": ["shaft", "tip"],
 "Dutch_oven": ["body", "lid", "handles"],
 "electric_guitar": ["body", "neck", "pickups", "bridge", "headstock"],
 "electric_locomotive": ["nose", "cab", "pantograph", "bogies"],
 "entertainment_center": ["shelves", "cabinet", "openings"],
 "face_powder": ["compact", "pad", "powder pan"],
 "feather_boa": ["feathers", "core"],
 "file": ["blade", "handle"],
 "fireboat": ["hull", "deck", "water monitors", "superstructure"],
 "fire_engine": ["cab", "pump panel", "ladders", "hose"],
 "fire_screen": ["frame", "mesh panels", "stand"],
 "flagpole": ["pole", "finial", "halyard", "cleat"],
 "flute": ["body", "headjoint", "keys", "footjoint"],
 "folding_chair": ["seat", "back", "frame", "hinge"],
 "football_helmet": ["shell", "face mask", "chin strap", "padding"],
 "forklift": ["mast", "forks", "cab", "counterweight"],
 "fountain": ["bowl", "nozzle", "basin"],
 "fountain_pen": ["nib", "barrel", "cap", "clip"],
 "four-poster": ["posts", "frame", "canopy", "headboard"],
 "freight_car": ["body", "doors", "bogies"],
 "French_horn": ["bell", "leadpipe", "rotary valves"],
 "fur_coat": ["body", "collar", "sleeves", "lining"],
 "garbage_truck": ["cab", "hopper", "lift mechanism", "wheels"],
 "gasmask": ["facepiece", "filters", "straps"],
 "goblet": ["cup", "stem", "base"],
 "go-kart": ["chassis", "steering wheel", "wheels", "seat"],
 "golf_ball": ["surface", "dimples"],
 "golfcart": ["body", "seats", "steering", "wheels"],
 "gondola": ["hull", "seat", "oarlock"],
 "gong": ["disk", "suspension", "mallet"],
 "grand_piano": ["lid", "keyboard", "strings", "legs"],
 "greenhouse": ["glazing", "frame", "benches", "vent"],
 "grocery_store": ["shelves", "checkout", "produce display"],
 "guillotine": ["frame", "blade", "lunette"],
 "hair_slide": ["body", "clip"],
 "hair_spray": ["can", "nozzle", "cap"],
 "half_track": ["cab", "tracks", "wheels", "body"],
 "hand-held_computer": ["screen", "buttons", "housing"],
 "handkerchief": ["cloth", "hem"],
 "hard_disc": ["platter", "spindle", "housing"],
 "harmonica": ["cover plates", "comb", "reeds"],
 "harp": ["soundboard", "neck", "strings", "pedals"],
 "harvester": ["header", "body", "wheels", "grain tank"],
 "hatchet": ["head", "handle"],
 "holster": ["pouch", "belt loop", "retention strap"],
 "honeycomb": ["cells", "frame"],
 "hook": ["shank", "point", "eye"],
 "hoopskirt": ["hoops", "fabric", "waistband"],
 "horizontal_bar": ["bar", "supports", "base"],
 "horse_cart": ["body", "wheels", "shaft", "seat"],
 "hourglass": ["upper bulb", "lower bulb", "narrow waist"],
 "iPod": ["screen", "wheel", "body", "case"],
 "jack-o-lantern": ["hollowed shell", "carved face", "handle"],
 "jeep": ["grille", "windshield", "doors", "wheels"],
 "jigsaw_puzzle": ["pieces", "image surface"],
 "jinrikisha": ["shaft", "seat", "wheels", "handles"],
 "joystick": ["stick", "base", "buttons"],
 "knot": ["loop", "bight", "tail"],
 "lab_coat": ["body", "sleeves", "pockets", "collar"],
 "lawn_mower": ["deck", "blade", "handle", "wheels"],
 "lens_cap": ["cap body", "retention strap"],
 "library": ["shelves", "reading tables", "stacks"],
 "lifeboat": ["hull", "oars", "seats", "lifelines"],
 "limousine": ["body", "windows", "doors", "wheels"],
 "liner": ["hull", "deck", "superstructure"],
 "lotion": ["bottle", "pump", "cap"],
 "loupe": ["lens", "frame", "handle"],
 "lumbermill": ["saw", "conveyor", "log deck"],
 "magnetic_compass": ["card", "housing", "needle", "base"],
 "mailbag": ["body", "strap", "closure"],
 "mailbox": ["box", "door", "flag"],
 "maillot_jersey": ["body", "straps", "leg openings"],
 "maillot_swimsuit": ["body", "straps", "leg openings"],
 "manhole_cover": ["cover", "rim"],
 "maraca": ["body", "handle"],
 "marimba": ["keys", "resonators", "frame", "mallets"],
 "mask": ["face piece", "straps", "eye openings"],
 "maypole": ["pole", "ribbons", "base"],
 "maze": ["paths", "walls", "entrance"],
 "medicine_chest": ["door", "shelves", "mirror"],
 "megalith": ["stone block", "capstone"],
 "microphone": ["head", "body", "stand mount"],
 "military_uniform": ["jacket", "trousers", "insignia", "buttons"],
 "milk_can": ["body", "lid", "handle"],
 "minibus": ["body", "doors", "windows", "wheels"],
 "miniskirt": ["waistband", "skirt body", "hem"],
 "minivan": ["body", "sliding door", "windows", "wheels"],
 "missile": ["warhead", "body", "fins"],
 "mobile_home": ["body", "windows", "door", "skirting"],
 "Model_T": ["body", "wheels", "roof", "windshield"],
 "modem": ["ports", "indicator lights", "case"],
 "monastery": ["cloister", "church", "cells"],
 "moped": ["frame", "engine", "seat", "wheels"],
 "mortar": ["barrel", "base plate", "breech"],
 "mortarboard": ["board", "tassel", "band"],
 "mosque": ["dome", "minaret", "prayer hall", "entrance"],
 "mosquito_net": ["mesh", "frame", "suspension"],
 "motor_scooter": ["deck", "handlebar", "wheels", "seat"],
 "mountain_bike": ["frame", "fork", "wheels", "handlebar"],
 "mousetrap": ["base", "spring", "bar", "trigger"],
 "moving_van": ["box", "cab", "door", "lift"],
 "neck_brace": ["support band", "chin rest", "straps"],
 "nipple": ["teat", "base"],
 "notebook": ["cover", "pages", "binding"],
 "obelisk": ["shaft", "base", "capstone"],
 "oboe": ["body", "keys", "reed"],
 "ocarina": ["body", "mouthpiece", "finger holes"],
 "odometer": ["display", "gear housing"],
 "oil_filter": ["canister", "filter media", "gasket"],
 "organ": ["pipes", "keyboard", "pedals", "case"],
 "oscilloscope": ["screen", "controls", "case"],
 "overskirt": ["outer skirt", "waistband", "hem"],
 "oxcart": ["bed", "wheels", "yoke"],
 "oxygen_mask": ["facepiece", "straps", "hose"],
 "packet": ["wrapper", "seal"],
 "paddle": ["shaft", "blade", "grip"],
 "paddlewheel": ["paddles", "axle", "housing"],
 "pajama": ["top", "bottom", "waistband"],
 "palace": ["facade", "towers", "gate", "courtyard"],
 "panpipe": ["pipes", "binding"],
 "parachute": ["canopy", "suspension lines", "harness"],
 "parallel_bars": ["bars", "supports", "base"],
 "parking_meter": ["head", "dial", "post"],
 "passenger_car": ["body", "doors", "windows", "wheels"],
 "pay-phone": ["handset", "coin slot", "dial", "housing"],
 "pedestal": ["platform", "shaft", "base"],
 "Petri_dish": ["dish", "lid"],
 "photocopier": ["feeder", "glass platen", "control panel", "output tray"],
 "pick": ["tip", "shaft", "handle"],
 "pickelhaube": ["helmet shell", "spike", "chin strap"],
 "picket_fence": ["posts", "slats", "rails"],
 "pickup_truck": ["cab", "bed", "wheels", "grille"],
 "pier": ["deck", "piles", "railings"],
 "piggy_bank": ["body", "slot", "snout", "feet"],
 "ping-pong_ball": ["surface"],
 "pinwheel": ["blades", "hub", "stick"],
 "pirate": ["hat", "coat", "sash", "boot"],
 "plane": ["nose", "wings", "fuselage", "tail"],
 "planetarium": ["dome", "projector", "seating"],
 "plow": ["blade", "frame", "hitch", "wheels"],
 "Polaroid_camera": ["body", "lens", "viewfinder", "film slot"],
 "pole": ["shaft", "base", "top"],
 "police_van": ["cab", "rear compartment", "lights", "wheels"],
 "poncho": ["cape", "neck opening", "hem"],
 "pool_table": ["playing surface", "rails", "pockets", "legs"],
 "pot": ["rim", "body", "handle", "base"],
 "potter_wheel": ["head", "foot pedal", "wheel"],
 "prayer_rug": ["field", "border", "fringe"],
 "prison": ["walls", "towers", "gates", "cells"],
 "missile_projectile": ["nose", "body", "tail"],
 "projector": ["lens", "body", "mount", "controls"],
 "puck": ["disc"],
 "purse": ["body", "strap", "closure", "pocket"],
 "quill": ["shaft", "barbs", "tip"],
 "quilt": ["patches", "binding", "stitch lines"],
 "racer": ["cockpit", "body", "wheels", "wing"],
 "radiator": ["fins", "inlet", "outlet", "core"],
 "radio": ["speaker", "tuning dial", "antenna", "housing"],
 "radio_telescope": ["dish", "feed", "support structure"],
 "rain_barrel": ["body", "inlet", "spigot", "lid"],
 "recreational_vehicle": ["cab", "living area", "windows", "wheels"],
 "reel": ["spool", "handle", "frame"],
 "reflex_camera": ["body", "lens", "viewfinder", "shutter"],
 "refrigerator": ["door", "shelves", "handle", "compressor"],
 "restaurant": ["entrance", "dining area", "counter", "kitchen view"],
 "revolver": ["barrel", "cylinder", "grip", "trigger"],
 "rifle": ["barrel", "stock", "sight", "trigger"],
 "rocking_chair": ["seat", "back", "rockers", "armrests"],
 "rotisserie": ["spit", "motor", "frame", "drip tray"],
 "rubber_eraser": ["body", "edge"],
 "rugby_ball": ["panel", "seams"],
 "safe": ["door", "dial", "body", "hinge"],
 "sarong": ["wrap", "waist tie", "hem"],
 "sax": ["body", "neck", "mouthpiece", "keys"],
 "school_bus": ["front", "windows", "doors", "wheels"],
 "schooner": ["hull", "masts", "sails", "deck"],
 "screen": ["frame", "mesh", "mount"],
 "screwdriver": ["shaft", "tip", "handle"],
 "sewing_machine": ["needle area", "bed", "hand wheel", "foot pedal"],
 "shield": ["face", "rim", "handle"],
 "shoe_shop": ["shelves", "display", "counter"],
 "shoji": ["frame", "paper panels", "sliding track"],
 "shopping_cart": ["basket", "handle", "wheels"],
 "shower_curtain": ["curtain panel", "grommets", "rod"],
 "ski": ["tip", "camber", "binding", "tail"],
 "ski_mask": ["face opening", "eye openings", "edge"],
 "slide_rule": ["body", "slider", "scales"],
 "sliding_door": ["panel", "track", "handle"],
 "slot": ["opening", "face", "controls"],
 "snowmobile": ["front ski", "track", "seat", "handlebar"],
 "snowplow": ["blade", "frame", "hydraulics", "mount"],
 "soccer_ball": ["panels", "seams"],
 "solar_dish": ["dish", "receiver", "mount"],
 "sombrero": ["crown", "brim", "band"],
 "space_bar": ["keycap"],
 "space_shuttle": ["nose", "fuselage", "wings", "tail"],
 "speedboat": ["hull", "deck", "windshield", "engine"],
 "spider_web": ["radial threads", "spiral threads", "anchor points"],
 "spindle": ["shaft", "whorl", "hook"],
 "sports_car": ["body", "windshield", "wheels", "spoiler"],
 "spotlight": ["lamp", "reflector", "housing", "mount"],
 "stage": ["platform", "backdrop", "human", "lighting"],
 "steam_locomotive": ["boiler", "cab", "tender", "wheels"],
 "steel_arch_bridge": ["arch", "deck", "support piers"],
 "steel_drum": ["body", "lip", "notes area"],
 "stethoscope": ["chest piece", "tubing", "earpieces"],
 "stole": ["strip", "ends"],
 "stone_wall": ["stones", "mortar", "cap"],
 "stopwatch": ["face", "buttons", "case"],
 "stove": ["burners", "control knobs", "oven door"],
 "streetcar": ["body", "windows", "doors", "wheels"],
 "stretcher": ["frame", "canvas", "handles", "wheels"],
 "studio_couch": ["seat", "back", "arms", "legs"],
 "stupa": ["dome", "harmika", "base"],
 "submarine": ["conning tower", "hull", "propeller", "rudder"],
 "sundial": ["dial plate", "gnomon", "base"],
 "sunglass": ["lens", "frame", "temples"],
 "sunscreen": ["tube", "cap", "dispensing nozzle"],
 "suspension_bridge": ["towers", "cables", "deck"],
 "swing": ["seat", "chains", "support"],
 "switch": ["toggle", "plate", "mount"],
 "tank": ["turret", "body", "tracks", "gun barrel"],
 "tape_player": ["reels", "controls", "play head"],
 "tennis_ball": ["surface"],
 "thatch": ["bundles", "ridge", "eaves"],
 "theater_curtain": ["curtain panel", "hem", "pelmet"],
 "thimble": ["cap", "rim"],
 "thresher": ["drum", "feeder", "frame"],
 "throne": ["seat", "back", "armrests", "base"],
 "tile_roof": ["tiles", "ridge", "eaves"],
 "tobacco_shop": ["shelves", "display", "counter"],
 "toilet_seat": ["seat", "lid", "hinges"],
 "torch": ["head", "body", "switch", "lens"],
 "tow_truck": ["cab", "boom", "wheels", "winch"],
 "toyshop": ["shelves", "display", "counter"],
 "tractor": ["body", "wheels", "cab", "three point hitch"],
 "trailer_truck": ["cab", "trailer", "wheels", "doors"],
 "trench_coat": ["collar", "buttons", "belt", "sleeves"],
 "tricycle": ["frame", "front wheel", "rear wheels", "seat"],
 "trimaran": ["center hull", "amas", "deck", "mast"],
 "tripod": ["legs", "head", "mount"],
 "triumphal_arch": ["archway", "column", "entablature"],
 "trombone": ["slide", "bell", "mouthpiece"],
 "tub": ["rim", "bowl", "drain"],
 "turnstile": ["arms", "shaft", "housing"],
 "typewriter_keyboard": ["keycaps", "carriage", "space bar"],
 "unicycle": ["wheel", "seat", "pedal"],
 "upright_piano": ["body", "keyboard", "strings", "pedal"],
 "vault": ["door", "chamber", "locking mechanism"],
 "velvet": ["pile", "edge"],
 "vending_machine": ["front panel", "selection buttons", "dispense slot"],
 "viaduct": ["arches", "deck", "piers"],
 "violin": ["body", "neck", "fingerboard", "strings", "bridge"],
 "volleyball": ["panels", "seams"],
 "waffle_iron": ["plates", "hinge", "handle"],
 "wall_clock": ["face", "hands", "case"],
 "wardrobe": ["doors", "rails", "drawers", "top"],
 "warplane": ["nose", "wings", "fuselage", "engines"],
 "washbasin": ["bowl", "tap", "overflow"],
 "washer": ["drum", "door", "control panel"],
 "water_jug": ["neck", "body", "handle", "base"],
 "water_tower": ["tank", "support", "ladder"],
 "wig": ["cap", "hair"],
 "window_screen": ["frame", "mesh", "spline"],
 "window_shade": ["shade panel", "roller", "cord"],
 "Windsor_tie": ["knot", "blade", "tail"],
 "wing": ["airfoil", "flap", "slat"],
 "wool": ["fibre", "bundle"],
 "worm_fence": ["posts", "rails", "weave"],
 "wreck": ["bow", "stern", "hull"],
 "yawl": ["hull", "mast", "sails", "rudder"],
 "yurt": ["wall", "roof ring", "cover", "door"],
 "web_site": ["header", "navigation", "content area", "footer"],
 "crossword_puzzle": ["grid", "clues", "filled squares"],
 "street_sign": ["post", "sign panel", "mount"],
 "traffic_light": ["lanes", "lights", "housing"],
 "menu": ["cover", "pages", "items"],
 "guacamole": ["bowl", "dip"],
 "consomme": ["bowl", "liquid"],
 "hot_pot": ["pot", "broth", "ingredients"],
 "trifle": ["layers", "glass", "topping"],
 "ice_cream": ["scoop", "cone", "topping"],
 "bagel": ["crust", "hole", "surface"],
 "pretzel": ["twist", "surface"],
 "cheeseburger": ["bun", "patty", "cheese", "toppings"],
 "hotdog": ["bun", "sausage", "toppings"],
 "mashed_potato": ["pile", "scoop"],
 "head_cabbage": ["outer leaves", "core", "head"],
 "broccoli": ["head", "stalk", "florets"],
 "cauliflower": ["curd", "stem", "florets"],
 "zucchini": ["stem end", "body", "blossom end"],
 "spaghetti_squash": ["skin", "flesh", "seeds"],
 "acorn_squash": ["skin", "flesh", "stem"],
 "butternut_squash": ["skin", "flesh", "stem"],
 "cucumber": ["stem end", "body", "blossom end"],
 "artichoke": ["head", "bracts", "stem"],
 "bell_pepper": ["stem", "body", "core"],
 "cardoon": ["stem", "leaves"],
 "mushroom": ["cap", "gills", "stem"],
 "Granny_Smith": ["skin", "flesh", "stem"],
 "strawberry": ["body", "calyx", "seeds"],
 "fig": ["skin", "flesh", "seeds"],
 "pineapple": ["crown", "skin", "core"],
 "jackfruit": ["skin", "flesh pods", "seeds"],
 "custard_apple": ["skin", "segments", "seeds"],
 "pomegranate": ["crown", "arils", "rind"],
 "hay": ["stalks", "bales"],
 "chocolate_sauce": ["bottle", "pour stream"],
 "dough": ["surface", "edges"],
 "meat_loaf": ["slice", "crust"],
 "pizza": ["crust", "cheese", "toppings", "slice"],
 "potpie": ["crust", "filling"],
 "burrito": ["tortilla", "filling", "ends"],
 "red_wine": ["glass", "liquid"],
 "espresso": ["cup", "crema"],
 "eggnog": ["cup", "foam"],
 "alp": ["peak", "slope", "snow cap"],
 "bubble": ["film", "reflection"],
 "cliff": ["face", "edge", "base"],
 "coral_reef": ["reef crest", "patches", "channels"],
 "geyser": ["vent", "plume", "pool"],
 "lakeside": ["shore", "water", "vegetation"],
 "promontory": ["headland", "cliff", "shoreline"],
 "sandbar": ["crest", "sand", "shallow water"],
 "seashore": ["beach", "waves", "shoreline"],
 "valley": ["floor", "sides", "stream"],
 "volcano": ["cone", "crater", "lava flow"],
 "ballplayer": ["helmet", "jersey", "glove"],
 "groom": ["suit", "tie", "bouquet"],
 "scuba_diver": ["mask", "tank", "fins", "wetsuit"],
 "rapeseed": ["flowers", "stalks", "pods"],
 "daisy": ["disk", "petals", "stem"],
 "yellow_lady_slipper": ["pouch", "sepals", "stem"],
 "corn": ["ear", "kernels", "husks"],
 "acorn": ["cap", "nut"],
 "hip": ["fruit", "sepals"],
 "buckeye": ["shell", "nut"],
 "coral_fungus": ["branches", "tips"],
 "agaric": ["cap", "gills", "stem"],
 "gyromitra": ["cap", "stem"],
 "stinkhorn": ["stalk", "cap", "gleba"],
 "earthstar": ["central disc", "rays"],
 "hen-of-the-woods": ["clusters", "caps"],
 "bolete": ["cap", "pore surface", "stem"],
 "corn_ear": ["kernels", "cob", "husk", "lobe", "corn"]
}

ALL_LABEL = list(PART_LABEL.keys())
ALL_LABEL.sort()


def build_index_mapping():
    starts = {};i = 0
    for k, v in PART_LABEL.items():
        starts[k] = (i, i + len(v))
        i += len(v)

    mapping = {}
    for idx, k in enumerate(ALL_LABEL):
        s, e = starts.get(k)
        mapping[idx] = (s, e)
    return mapping
CLASS2PART_MAPPING = build_index_mapping()


MEAN = {'cifar10': [0.4914, 0.4822, 0.4465],
        'cifar100': [0.507, 0.487, 0.441],
        'imagenet100': [0.485, 0.456, 0.406],
        'domainnet': [0.6023, 0.5827, 0.5495],
        'domainnetl': [0.6023, 0.5827, 0.5495],
        'domainnets': [0.9507, 0.9507, 0.9507],
        'clip': [0.48145466, 0.4578275, 0.40821073]}
STD = {'cifar10': [0.2470, 0.2435, 0.2616],
       'cifar100': [0.267, 0.256, 0.276],
       'imagenet100': [0.229, 0.224, 0.225],
       'domainnet': [0.3246, 0.3194, 0.3419],
       'domainnetl': [0.3246, 0.3194, 0.3419],
       'domainnets': [0.1060, 0.1060, 0.1060],
       'clip': [0.26862954, 0.26130258, 0.27577711]}

NUM_CLASSES = {'cifar10': 10,
               'cifar100': 100,
               'imagenet100': 100,
               'imagenet1000': 1000,
               'imagenet54': 54,
               'imagenet104': 104,
               'imagenet896': 896,
               'domainnet': 100,
               'domainnetl': 100,
               'domainnets': 100,
               'rot': 4}
M_IN = {'cifar10': 23,
        'cifar100': 27,
        'imagenet100': 25,
        'imagenet1000': 25,
        'domainnet': 25,
        'domainnetl': 25,
        'domainnets': 25}

n = 50
METRIC_OOD = roc_auc_score

def get_seed():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(), 
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

def setup_seed(seed=None, state=None):
    if state is None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_exist_and_skip(ckpt_path, savemark):
    ckp_path_sp = ckpt_path + savemark + '.pth'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if len(glob(ckp_path_sp[:-4] + '*' + ckp_path_sp[-4:])) != 0 and 'nosave' not in ckpt_path:
        while True:
            a = input(ckp_path_sp + ' existed, overwrite it? [Y] / N')
            if a in ['y', 'Y', '']:
                print('overwrite:', ckp_path_sp)
                return False
            elif a in ['n', 'N', ]:
                return True
    else:
        print('save to:', ckp_path_sp)
        return False

keras_to_synset = {name: synset_id for synset_id, name in IMAGENET_LABEL.values()}
def revise_p(ordered, classnames_path='classnames.txt'):
    synset_to_imagenetstd = {}
    with open(classnames_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ctx = line.split()
            synset_to_imagenetstd[ctx[0]] = ' '.join(ctx[1:])
            
    revised = []
    for keras_name in ordered:
        synset_id = keras_to_synset.get(keras_name)
        if synset_id is None:
            raise KeyError(f"Keras name '{keras_name}' not found in IMAGENET_LABEL")
        std_name = synset_to_imagenetstd.get(synset_id)
        if std_name is None:
            raise KeyError(f"Synset ID '{synset_id}' not found in classnames.txt")
        revised.append(std_name)
    return revised

def get_text_from_datapath_withref(label):
    fixedkey = revise_p(label.keys())
    parttext = [k+"'s "+v_ for k, v in zip(fixedkey, label.values()) for v_ in v]
    orilabel = revise_p(ALL_LABEL)
    return orilabel, parttext

def get_text_from_datapath(p, label=None, with_cls=True):
    if with_cls:
        parttext = [k+"'s "+v_ for k, v in (PART_LABEL if label is None else label).items() for v_ in v]
    else:
        parttext = [v_ for v in (PART_LABEL if label is None else label).values() for v_ in v]
    
    return ALL_LABEL, parttext


def balanced_sampling(label: torch.LongTensor, need: int):
    inverse = torch.unique(label, return_inverse=True)[1]
    counts = torch.bincount(inverse)
    weights = 1.0 / counts[inverse].to(dtype=torch.float32)
    weights = weights / weights.sum()

    sampled_idx = torch.multinomial(weights, need, replacement=False)
    return sampled_idx



def fit_ransac(pts_src, pts_dst, pts_label, transform_mode='affine'):
    device = pts_src.device
    N = pts_src.shape[0]
    if N == 0:
        return None, torch.zeros((0,), dtype=torch.bool, device=device), torch.zeros((0,), device=device)

    best_M, best_mean, best_cnt = None, float('inf'), 0
    for _ in range(500):
      if N < 3: break
      sample_idx = balanced_sampling(pts_label, 3)
      M = solve_affine(pts_src[sample_idx], pts_dst[sample_idx])
      if M is None: continue
      trans = transform_points_affine(M, pts_src)
      residuals = (trans - pts_dst).norm(dim=1)
      inliers = residuals <= 4
      cnt = int(inliers.sum().item())
      if cnt >= (3 if transform_mode=='affine' else 4):
        mean_in = float(residuals[inliers].mean().item())
        if (cnt > best_cnt) or (cnt == best_cnt and mean_in < best_mean):
          best_cnt, best_mean = cnt, mean_in
          model_ref = (solve_affine if (transform_mode=='affine' or cnt<4 and transform_mode=='homography') else solve_homography)(pts_src[inliers], pts_dst[inliers])
          best_M = model_ref if model_ref is not None else M

    if best_M is None:
      best_M = (solve_affine if transform_mode=='affine' else solve_homography)(pts_src, pts_dst)
      if best_M is None:
        return None, torch.zeros((N,), dtype=torch.bool, device=device), torch.full((N,), float('inf'), device=device)
    trans = transform_points_affine(best_M, pts_src) if transform_mode=='affine' else transform_points_homography(best_M, pts_src)
    residuals = (trans - pts_dst).norm(dim=1)
    inliers = residuals <= 4
    return best_M, inliers, residuals


# def transform_points_affine_batch(M, pts):
    
#     A = M[:, :2, :2]   # (K,2,2)
#     t = M[:, :2, 2]    # (K,2)

#     # (K,N,2) = (K,N,2) @ (K,2,2)^T
#     out = torch.einsum('nd,kcd->knc', pts, A)

#     out = out + t[:, None, :]

#     return out


# def solve_affine_batch(pts_src_batch, pts_dst_batch):
#     K = pts_src_batch.shape[0]
#     device = pts_src_batch.device
#     dtype = pts_src_batch.dtype

#     x = pts_src_batch[:, :, 0]
#     y = pts_src_batch[:, :, 1]

#     u = pts_dst_batch[:, :, 0]
#     v = pts_dst_batch[:, :, 1]

#     A = torch.zeros((K, 6, 6), device=device, dtype=dtype)
#     b = torch.zeros((K, 6), device=device, dtype=dtype)

#     A[:, 0::2, 0] = x
#     A[:, 0::2, 1] = y
#     A[:, 0::2, 2] = 1

#     A[:, 1::2, 3] = x
#     A[:, 1::2, 4] = y
#     A[:, 1::2, 5] = 1

#     b[:, 0::2] = u
#     b[:, 1::2] = v

#     sol = torch.linalg.solve(A.float(), b.float())
#     # sol = torch.linalg.lstsq(
#     #     A.float(),
#     #     b.float()
#     # ).solution.squeeze(-1)

#     M = torch.zeros((K, 3, 3), device=device, dtype=dtype)
#     M[:, :2, :] = sol.view(K, 2, 3)
#     M[:, 2, 2] = 1

#     return M

# def batch_balanced_sampling(label, K, need=3):
#     inverse = torch.unique(label, return_inverse=True)[1]
#     counts = torch.bincount(inverse)

#     weights = 1.0 / counts[inverse].float()
#     weights = weights / weights.sum()

#     idx = torch.multinomial(weights,K * need,replacement=True
#     )

#     return idx.view(K, need)


# def non_collinear_mask(pts, eps=1e-6):
#     p1 = pts[:, 0]
#     p2 = pts[:, 1]
#     p3 = pts[:, 2]

#     area = ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])).abs()

#     return area > eps


# def fit_ransac_fast(
#     pts_src,
#     pts_dst,
#     pts_label,
# ):
#     samples = batch_balanced_sampling(pts_label,3
#     )
#     src_samples = pts_src[samples]
#     valid = non_collinear_mask(src_samples)
#     src_samples = src_samples[valid]
#     dst_samples = pts_dst[samples][valid]

#     M_all = solve_affine_batch(
#         src_samples,
#         dst_samples
#     )

#     pred = transform_points_affine_batch(M_all, pts_src)

#     residuals = (pred - pts_dst[None]).norm(dim=-1)

#     inliers = residuals < 4
#     counts = inliers.sum(1)

#     best_idx = counts.argmax()

#     return (
#         M_all[best_idx],
#         inliers[best_idx],
#         residuals[best_idx]
#     )
    
# def fit_ransac_cuda(
#     pts_src,
#     pts_dst,
#     pts_label,
# ):
#     device = pts_src.device
#     N = pts_src.shape[0]

#     if N < 3:
#         return None, None, None

#     samples = batch_balanced_sampling(
#     pts_label,
#     3
# )

#     src_samples = pts_src[samples]   # (K,3,2)
#     valid = non_collinear_mask(src_samples)
#     src_samples = src_samples[valid]
#     dst_samples = pts_dst[samples][valid]
#     # dst_samples = pts_dst[samples]
    
    

#     M_all = solve_affine_batch(src_samples, dst_samples)   # (K,3,3)

#     pred_all = transform_points_affine_batch(M_all, pts_src)   # (K,N,2)

#     residuals = (pred_all - pts_dst[None]).norm(dim=-1)   # (K,N)

#     inliers = residuals <= 4
#     counts = inliers.sum(dim=1)

#     means = torch.where(
#         counts > 0,
#         residuals.masked_fill(~inliers, 0).sum(dim=1) / counts.clamp(min=1),
#         torch.full_like(counts, float('inf'), dtype=torch.float)
#     )
#     score = counts.float() * 1e6 - means

#     best_idx = score.argmax()

#     best_M = M_all[best_idx]
#     best_inliers = inliers[best_idx]
#     best_residuals = residuals[best_idx]

#     return best_M, best_inliers, best_residuals


# from concurrent.futures import ProcessPoolExecutor
# def _ransac_iter(args):
#     pts_src, pts_dst, pts_label, transform_mode, need = args

#     N = pts_src.shape[0]
#     sample_idx = balanced_sampling(pts_label, need)

#     if need == 3:
#         M = solve_affine(pts_src[sample_idx], pts_dst[sample_idx])
#     else:
#         M = solve_homography(pts_src[sample_idx], pts_dst[sample_idx])

#     if M is None:
#         return None, 0, float('inf')

#     if need == 3:
#         trans = transform_points_affine(M, pts_src)
#     else:
#         trans = transform_points_homography(M, pts_src)

#     residuals = (trans - pts_dst).norm(dim=1)
#     inliers = residuals <= 4
#     cnt = int(inliers.sum().item())

#     if cnt < need:
#         return None, cnt, float('inf')

#     mean_in = float(residuals[inliers].mean().item())
#     return M, cnt, mean_in

# def fit_ransac_mp(pts_src, pts_dst, pts_label,
#                   transform_mode='affine',
#                   mp=4):

#     device = pts_src.device
#     N = pts_src.shape[0]
#     if N == 0:
#         return None, torch.zeros((0,), dtype=torch.bool, device=device), torch.zeros((0,), device=device)

#     need = 3 if transform_mode=='affine' else (4 if N>=4 else 3)

#     jobs = [(pts_src, pts_dst, pts_label, transform_mode, need)
#             for _ in range(500)]

#     if mp > 1:
#         with ProcessPoolExecutor(max_workers=mp) as ex:
#             results = list(ex.map(_ransac_iter, jobs))
#     else:
#         results = [_ransac_iter(job) for job in jobs]

#     best_M, best_cnt, best_mean = None, 0, float('inf')
#     for M, cnt, mean_in in results:
#         if M is None:
#             continue
#         if cnt > best_cnt or (cnt == best_cnt and mean_in < best_mean):
#             best_M, best_cnt, best_mean = M, cnt, mean_in

#     if best_M is None:
#         best_M = (solve_affine if transform_mode=='affine' else solve_homography)(pts_src, pts_dst)
#         if best_M is None:
#             return None, torch.zeros((N,), dtype=torch.bool, device=device), torch.full((N,), float('inf'), device=device)

#     trans = transform_points_affine(best_M, pts_src) if transform_mode=='affine' else transform_points_homography(best_M, pts_src)
#     residuals = (trans - pts_dst).norm(dim=1)
#     inliers = residuals <= 4
#     return best_M, inliers, residuals




def solve_affine(pts_src: torch.Tensor, pts_dst: torch.Tensor) -> Optional[torch.Tensor]:
    N = pts_src.shape[0]
    if N < 3:
        return None
    A = torch.zeros((2 * N, 6), dtype=pts_src.dtype, device=pts_src.device)
    b = torch.zeros((2 * N,), dtype=pts_src.dtype, device=pts_src.device)
    A[0::2, 0] = pts_src[:, 0];A[0::2, 1] = pts_src[:, 1];A[0::2, 2] = 1
    A[1::2, 3] = pts_src[:, 0];A[1::2, 4] = pts_src[:, 1];A[1::2, 5] = 1
    b[0::2] = pts_dst[:, 0];b[1::2] = pts_dst[:, 1]
    # x = torch.lstsq(b.unsqueeze(1), A)[0][:6].view(6)# if hasattr(torch, 'lstsq') else torch.linalg.lstsq(A, b).solution.unsqueeze(1)
    # try:
    sol = torch.linalg.lstsq(A.float(), b.unsqueeze(1).float()).solution
    return torch.cat([sol.squeeze(1).view(2, 3), torch.tensor([[0., 0., 1.]])], dim=0).to(pts_src.dtype).to(pts_src.device)


def solve_homography(pts_src: torch.Tensor, pts_dst: torch.Tensor) -> Optional[torch.Tensor]:
    # DLT: pts_src, pts_dst: (N,2) N>=4
    N = pts_src.shape[0]
    if N < 4:
        return None
    
    x, y = pts_src[:, 0], pts_src[:, 1];u, v = pts_dst[:, 0], pts_dst[:, 1]
    A = torch.zeros((2*N, 9), dtype=pts_src.dtype, device=pts_src.device)
    A[0::2, 0] = -x;   A[0::2, 1] = -y;  A[0::2, 2] = -1
    A[1::2, 3] = -x;   A[1::2, 4] = -y;  A[1::2, 5] = -1
    A[0::2, 6] = u * x; A[0::2, 7] = u * y; A[0::2, 8] = u
    A[1::2, 6] = v * x; A[1::2, 7] = v * y; A[1::2, 8] = v
    # A = []
    # for i in range(N):
    #     x, y = pts_src[i].tolist()
    #     u, v = pts_dst[i].tolist()
    #     A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    #     A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    # A = torch.tensor(A, dtype=pts_src.dtype, device=pts_src.device)  # (2N,9)
    # SVD
    H = torch.linalg.svd(A.float())[2][-1, :].view(3, 3)# torch.linalg.svd(A)[2][-1, :].view(3, 3)
    if abs(H[2, 2]) < EPS:
        return None
    return (H / H[2, 2]).to(pts_src.dtype).to(pts_src.device)

def transform_points_affine(M: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    K = pts.shape[0]
    homo = torch.cat([pts, torch.ones((K,1), device=pts.device, dtype=pts.dtype)], dim=1)
    return (M @ homo.T).T[:, :2]

def transform_points_homography(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    K = pts.shape[0]
    homo = torch.cat([pts, torch.ones((K,1), device=pts.device, dtype=pts.dtype)], dim=1)
    th = (H @ homo.T).T
    return th[:, :2] / th[:, 2:].clamp_min(EPS)

def fps_sampling_faiss(X: torch.Tensor, k: int,
                        already_selected: list = None,
                        mask: torch.Tensor = None):
    device = X.device
    N, D = X.shape
    mask = torch.ones(N, dtype=torch.bool, device=device) if mask is None else mask.to(device)
    mask_np = mask.cpu().numpy().astype(bool)
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= num_valid:
        return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.long).to(device)

    Xn = F.normalize(X.to(torch.float32).contiguous(), p=2, dim=1, eps=EPS)
    X_np = Xn.numpy().astype('float32') 

    index = faiss.IndexFlatIP(D)  #

    index.add(X_np)  
    def sims_of_query_all(q_np):
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        Dists, Ids = index.search(q_np, N) 
        sims = np.full(N, -np.inf, dtype='float32')
        sims[Ids[0]] = Dists[0]
        return sims

    S = Xn.sum(dim=0)  
    scores = Xn.matmul(S)
    scores[~mask.cpu()] = -float('inf')
    selected = [int(torch.argmax(scores).item())] if already_selected is None or len(already_selected)==0 else list(already_selected)

    candidates = np.ones(N, dtype=np.bool_)
    for s in selected:
        candidates[int(s)] = False
    candidates = candidates & mask_np  # 

    dist_np = np.full((N,), 1e9, dtype='float32')

    for s in selected:
        q = Xn[int(s)].numpy().astype('float32')
        sims = sims_of_query_all(q)
        sims[~mask_np] = -np.inf
        cur_d = 1.0 - sims
        dist_np = np.minimum(dist_np, cur_d)

    to_select = k - len(selected)
    for _ in range(to_select):
        if not candidates.any():
            break
        masked = dist_np.copy()
        masked[~candidates] = -1e9 
        nxt = int(np.argmax(masked))
        selected.append(nxt)
        candidates[nxt] = False

        # 更新 dist
        q = Xn[nxt].numpy().astype('float32')
        sims = sims_of_query_all(q)
        sims[~mask_np] = -np.inf
        cur_d = 1.0 - sims
        dist_np = np.minimum(dist_np, cur_d)

    sel = np.array(selected[:k], dtype=np.int64)
    return torch.from_numpy(sel).to(device)

def fps_sampling(X: torch.Tensor, k: int, already_selected: List[torch.Tensor]=None, mask: torch.Tensor=None, ):
    return fps_sampling_faiss(X, k, already_selected, mask)
    # X: (N,d) 
    X = l2_normalize(X, dim=-1)
    N = X.size(0)
    device = X.device
    mask = torch.ones(N, dtype=torch.bool, device=device) if mask is None else mask.to(device)
    num_valid = int(mask.sum().item())
    if num_valid == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= num_valid:
        return torch.nonzero(mask, as_tuple=False).squeeze(1).to(torch.long).to(device)
    
    selected = [torch.matmul(X, X.t()).sum(-1).masked_fill(~mask, -INF).argmax()] if already_selected is None or len(already_selected)==0 else list(already_selected)
    candidates = torch.ones(N, dtype=torch.bool, device=device)
    candidates[selected] = False
    candidates *= mask
    
    print('cur_dist')
    dist = torch.full((N,), INF, device=device)
    cur_dist = 1. - torch.matmul(X, X[selected].t()).amax(dim=1)
    dist = torch.minimum(dist, cur_dist)
    for _ in range(k):
        if not candidates.sum().item():
            break
        nxt = dist.masked_fill(~candidates, -INF).argmax().item().long()
        selected.append(nxt)
        candidates[nxt] = False
        cur_d = 1. - torch.matmul(X, X[nxt:nxt+1].t()).squeeze(-1)
        dist = torch.minimum(dist, cur_d)
    return selected[:k]

def get_fake_fc(feature, gt, C):
    D = feature.size(1)
    mu = torch.zeros(C, D)  
    for i in range(C):
        mu[i] = feature[gt == i].mean(dim=0)

    fc = nn.Linear(D, C).cuda()
    with torch.no_grad():
        fc.weight.copy_(F.normalize(mu, p=2, dim=1).cuda())
        fc.bias.zero_()

    return fc


def get_available_gpus(min_free_mem_gb=12, max_jobs=8):
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    free_mems = [int(x.strip()) for x in result.strip().split('\n')]  # MB
    gpu_ids = [i for i, mem in enumerate(free_mems) if mem > min_free_mem_gb * 1024]
    gpu_ids = [idx for idx in gpu_ids if idx not in [0,'0']]
    if len(gpu_ids)>max_jobs:
        gpu_ids.sort(key=lambda x:int(x))
        gpu_ids = gpu_ids[-max_jobs:]
    return gpu_ids



def fpr95(y_true, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true.astype(bool)[desc_score_indices]

    threshold_idxs = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    out = np.cumsum(y_true, dtype=np.float64)
    if not np.allclose(out[-1], np.sum(y_true, dtype=np.float64), rtol=1e-05, atol=1e-08):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    tps = out[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    sl = slice(tps.searchsorted(tps[-1]), None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    return fps[np.argmin(np.abs(recall - 0.95))] / (np.sum(np.logical_not(y_true)))


def get_measures(_pos, _neg):
    pos = np.array(_pos)
    neg = np.array(_neg)
    score = np.concatenate([pos, neg], axis=0)
    labels = np.array([1]*len(pos)+[0]*len(neg), dtype=np.int32)

    return roc_auc_score(labels, score), fpr95(labels, score)


def minmax(l, return_scaler=False):
    l = np.array(l)
    nancnt = np.isnan(l).sum()
    if nancnt:
        raise 'nan'
    if return_scaler:
        mean, std = l.min(), l.max() - l.min() + 1e-12
        return (l - mean) / std, (mean, std)
    return (l - l.min()) / (l.max() - l.min() + 1e-12)


def meanstd(l, scaler=None, toscaler=None, return_scaler=False):
    l = np.array(l)
    mean, std = np.mean(l), np.std(l) + 1e-12
    if scaler is not None:
        return (l - scaler[0]) / scaler[1]
    elif toscaler is not None:
        std = toscaler[1] / std
        return std * l + (toscaler[0] - mean * std)
    elif return_scaler:
        return (l - mean) / std, (mean, std)
    return (l - mean) / std

def meanstd_(l, l_, neg=True):
    mean, std = -np.array(l).mean() if neg else np.array(l).mean(), np.array(l).std() + 1e-12
    l = -np.array(l+l_) if neg else np.array(l+l_)
    return (l - mean) / std


def smooth_l1(diff, beta=1., reduction='none'):
    diff = diff.abs()
    loss = torch.where(diff < beta,
                       0.5 * (diff ** 2) / beta,
                       diff - 0.5 * beta)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()


def macro_AUC(gt, pred, split):
    return sum([roc_auc_score(y_true=np.append(gt[:split[0]], gt[split[i]:split[i + 1]]),
                              y_score=np.append(pred[:split[0]], pred[split[i]:split[i + 1]])) for i in
                range(len(split) - 1)]) / (len(split) - 1)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)

# def kls(p):
#     mask = p > 0
#     return -(p[mask] * p[mask].log()).sum()
def kls(p):
    return -(p * p.clamp_min(EPS).log()).sum(1)

def getdis(x, a, metric='euler'):
    if metric == 'euler':
        return F.relu(torch.sum(x ** 2, dim=1, keepdim=True) + \
                      torch.sum(a ** 2, dim=1) - \
                      2 * torch.matmul(x, a.t())) ** 0.5
    elif metric == 'cosine':
        return 1 - torch.matmul(F.normalize(x, dim=1), F.normalize(a, dim=1).t())


def knn_predicter(pred, clean, gt_clean, K=5):
    _, topk_cidx = torch.topk(getdis(pred, clean), K, dim=1, largest=False)  # B X K
    topk_cgt = gt_clean[topk_cidx.flatten()].view(topk_cidx.size())
    resc = []
    for i, cgt in enumerate(topk_cgt):
        elec, cntc = torch.unique(cgt, return_counts=True)
        if len(cntc) == 1:
            resc.append(topk_cgt[i, 0])
        else:
            vc, kc = torch.topk(cntc, 2)
            if vc[0] == vc[1]:
                resc.append(topk_cgt[i, 0])
            else:
                resc.append(elec[kc[0]])
    return torch.from_numpy(np.array(resc)).cuda()


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std