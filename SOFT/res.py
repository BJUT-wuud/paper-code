import torch


# tnews
# seed = 2, 1: 24875621890547264, 4: 39502487562189054, 8: 4955223880597015, 16: 5194029850746269
# seed = 144, 1: 17761194029850746, 4: 4154228855721393, 8: 43134328358208956, 16: 5154228855721393
# seed = 145, 1: 2064676616915423, 4: 2930348258706468 , 8: 43283582089552236, 16: 5203980099502488


####################################### tnews ##############################################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                64: 5691542288557214
shot1 = torch.Tensor([24.875621890547264, 17.761194029850746, 20.64676616915423])
shot4 = torch.Tensor([39.502487562189054, 41.54228855721393, 29.30348258706468])
shot8 = torch.Tensor([49.55223880597015, 43.134328358208956, 43.283582089552236])
shot16 = torch.Tensor([51.94029850746269, 51.54228855721393, 52.03980099502488])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))

####################################### cnews ##############################################
# cnews
# seed = 2, 1: 5028, 4: 7553 , 8: 8976, 16: 9422
# seed = 145, 1: 5445, 4: 7955 , 8: 9377, 16: 9423
# seed = 144, 1: 5695, 4: 8837 , 8: 9269, 16: 9397

shot1 = torch.Tensor([50.28, 54.45, 56.95])
shot4 = torch.Tensor([75.53, 79.55, 88.37])
shot8 = torch.Tensor([89.76, 93.77, 92.69])
shot16 = torch.Tensor([94.22, 94.23, 93.97])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
#######################################csldcp##############################################

# csldcp
# seed = 2, 1: 1709641255605381, 4: 38621076233183854 , 8: 5179372197309418, 16: 5246636771300448
# seed = 144, 1: 2040358744394619, 4: 4282511210762332 , 8: 5162556053811659, 16: 5392376681614349
# seed = 145, 1: 1446188340807175, 4: 413677130044843 , 8: 5263452914798207, 16: 5257847533632287

shot1 = torch.Tensor([17.09641255605381, 20.40358744394619, 14.46188340807175])
shot4 = torch.Tensor([38.621076233183854, 42.82511210762332 , 41.3677130044843])
shot8 = torch.Tensor([51.79372197309418,  51.62556053811659, 52.63452914798207])
shot16 = torch.Tensor([52.46636771300448, 53.92376681614349, 52.57847533632287])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
