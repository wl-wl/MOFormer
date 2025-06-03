from Bio.SeqUtils import ProtParam
from Bio.Seq import Seq
from modlamp.descriptors import GlobalDescriptor

peptide_sequence = "GIGKFLHSAKKFGKAFVGEIMNS"
print(len(peptide_sequence))

seq = Seq(peptide_sequence)


aac_count = ProtParam.ProteinAnalysis(str(seq)).count_amino_acids()
print("count_amino_acids:", aac_count)


aac = ProtParam.ProteinAnalysis(str(seq)).get_amino_acids_percent()
print("get_amino_acids_percent:", aac)


molecular_weight = ProtParam.ProteinAnalysis(str(seq)).molecular_weight()
print("molecular_weight:", molecular_weight)

aromaticity = ProtParam.ProteinAnalysis(str(seq)).aromaticity()
print("aromaticity:", aromaticity)

instability_index = ProtParam.ProteinAnalysis(str(seq)).instability_index()
print("instability_index:", instability_index)


flexibility = ProtParam.ProteinAnalysis(str(seq)).flexibility()
print("flexibility:", flexibility)


gravy = ProtParam.ProteinAnalysis(str(seq)).gravy()
print("gravy:", gravy)


param_dict = {
    "hydrophobicity": ["KyteDoolittle"],
    "flexibility": ["KarplusSchulz"],
    "surface_accessibility": ["Eisenberg"],
}
protein_scale = ProtParam.ProteinAnalysis(str(seq)).protein_scale(param_dict,window=9, edge=1.0)
print("protein_scale:", protein_scale)


isoelectric_point = ProtParam.ProteinAnalysis(str(seq)).isoelectric_point()
print("iselectric_point:", isoelectric_point)


charge_at_PH = ProtParam.ProteinAnalysis(str(seq)).charge_at_pH(pH=7.4)
print("charge_at_PH:", charge_at_PH)

secondary_structure_fraction = ProtParam.ProteinAnalysis(str(seq)).secondary_structure_fraction()
print("secondary_structure_fraction:", secondary_structure_fraction)


molar_extinction_coefficient = ProtParam.ProteinAnalysis(str(seq)).molar_extinction_coefficient()
print("molar_extinction_coefficient:", molar_extinction_coefficient)

print('-----------------------------------------------------------------------------------------')
print('1.Length', '2.MW', '3.Charge', '4.ChargeDensity', '5.pI', '6.InstabilityInd', '7.Aromaticity', '8.AliphaticInd', '9.BomanInd', '10.HydrophRatio')
desc_all= GlobalDescriptor(peptide_sequence)
desc_all.calculate_all()
desc_all = desc_all.descriptor
desc_all = desc_all.ravel()
# desc_all.save_descriptor('/path/to/outputfile.csv')
print('desc_all:',desc_all)


# seq_d=[]
#
#
# desc_charge_density = GlobalDescriptor(seq_d)
# desc_charge_density.charge_density()
# charge_density_together = desc_charge_density.descriptor
# charge_density_together = charge_density_together.ravel()
#
# desc_instability = GlobalDescriptor(seq_d)
# desc_instability.instability_index()
# instability_together = desc_instability.descriptor
# instability_together = instability_together.ravel()
# instability_together = -1. * instability_together
#
# desc_boman = GlobalDescriptor(seq_d)
# desc_boman.boman_index()
# boman_together = desc_boman.descriptor
# boman_together = boman_together.ravel()
# boman_together = -1. * boman_together
#
