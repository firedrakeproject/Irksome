from .ButcherTableaux import ButcherTableau
import numpy as np


pep425A = np.array(
    [[0, 0, 0, 0],
     [1/10, 0, 0, 0],
     [-35816/35721, 56795/35721, 0, 0],
     [11994761/5328000, -11002961/4420800, 215846127/181744000, 0]])
pep425b = np.array([-17/222, 6250/15657, 5250987/10382126, 4000/23307])
pep425c = np.array([0, 1/10, 19/20, 37/63])

pep526A = np.array(
    [[0, 0, 0, 0, 0],
     [0.193445628056365, 0, 0, 0, 0],
     [-0.090431947690469, 0.646659568003039, 0, 0, 0],
     [-0.059239621354435, 0.598571867726670, -0.010476084304794, 0, 0],
     [0.173154586278662, 0.043637751980064, 0.949323298732961, -0.262838451019868, 0]])
pep526b = np.array([0.054828314201395, 0.310080077556546, 0.531276882919990, -0.135494569336049, 0.239309294658118])
pep526c = [0, 0.193445628056365, 0.55622762031257, 0.528856162067441, 0.9032771859718189]

pep636A = np.array(
    [[0, 0, 0, 0, 0, 0],
     [0.12316523079127038, 0, 0, 0, 0, 0],
     [-0.53348119048187126, 1.1200645707708279, 0, 0, 0, 0],
     [0.35987162974687092, -0.17675778446586507, 0.7331973326225617, 0, 0, 0],
     [0.015700424346522388, 0.02862938097533644, -0.014047147149911631, -0.015653338246176568, 0, 0],
     [-1.9608805853984794, -0.82154709029385564, -0.0033631561953843502, 0.046367461001250457, 2.782035718578454, 0]])
pep636b = np.array([0.78642719559722885, 0.69510370728230297, 0.42190724518033551, 0.21262030193155254, -0.70167978222250704, -0.41437866776891263])
pep636c = np.array([0, 0.12316523079127044, 0.58658338028895673, 0.91631117790356775, 0.014629319925770667, 0.042612347691984923])

pep746A = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [-0.10731260966924323, 0, 0, 0, 0, 0, 0],
     [0.14772934954602848, -0.12537555684690285, 0, 0, 0, 0, 0],
     [0.7016079790308741, -0.75094597518803941, 0.76631666070124027, 0, 0, 0, 0],
     [-0.8967481787471202, -0.43795858531068965, 1.7727346351832869, 0.1706052810617312, 0, 0, 0],
     [1.6243872270239892, -0.69700589895015241, -0.3861309831750398, -0.032848941899304235, 0.30227620385295728, 0, 0],
     [-0.32463926305048885, -0.3480143346241919, 1.3500419757109139, 0.039096802121597336, -0.17851883247877129, 0.010142489530892661, 0]])
pep746b = np.array([-0.69203318482299292, 0.0074442860308153933, 0.93216717844052677, -1.159431111205361, 0.27787978605406632, 0.93890392164164138, 0.69506912386130404])
pep746c = np.array([0, -0.10731260966924323, 0.022353792699125609, 0.71697866454407488, 0.60863315218720804, 0.81067760685245005, 0.54810883720995185])

pep756A = np.array(
    [[0, 0, 0, 0, 0, 0, 0],
     [0.34288981581855521, 0, 0, 0, 0, 0, 0],
     [0.16800230418143236, 0.1262987524809161, 0, 0, 0, 0, 0],
     [0.4326925567104672, -0.24221982610439177, 0.15241708521248304, 0, 0, 0, 0],
     [0.019843989305203335, 0.20330206481276515, -0.3494376489494413, 0.09780248603799992, 0, 0, 0],
     [3.5441758455721732, 9.884560134482289, -3.7993663287883006, -6.07804112569088, -2.820029405964353, 0, 0],
     [-16.625817935606782, -49.999620978741511, 22.3661445506308, 30.50526767511958, 13.408435545803448, 1.3455911427944685, 0]])
pep756b = np.array([0.15881394125505754, 3.390357323579911e-13, 0.4109696726168125, -1.6409254928717294e-13, -0.056173857997504642, 0.40542999348169673, 0.08096025064376304])
pep756c = np.array([0, 0.34288981581855521, 0.2943010566234846, 0.34288981581855849, -0.028489108793472939, 0.73129911961092908, 1.0000000000000007])

pepdict = {
    (4, 2, 5): (pep425A, pep425b, pep425c),
    (5, 2, 6): (pep526A, pep526b, pep526c),
    (6, 3, 6): (pep636A, pep636b, pep636c),
    (7, 4, 6): (pep746A, pep746b, pep746c),
    (7, 5, 6): (pep756A, pep756b, pep756c)
}


class PEPRK(ButcherTableau):
    def __init__(self, ns, order, peporder):
        try:
            A, b, c = pepdict[ns, order, peporder]
        except KeyError:
            raise NotImplementedError("No PEP method for that combination of stages, order and pseudo-energy preserving order")
        self.peporder = peporder
        super(PEPRK, self).__init__(A, b, None, c, order, None, None)