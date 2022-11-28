import torch


# tnews
# seed = 2, shot = 1, acc = 3870646766169154
# seed = 2, shot = 4, acc = 49651741293532337
# seed = 2, shot = 8, acc = 5303482587064676
# seed = 2, shot = 16, acc = 5522388059701493
# seed = 144, shot = 1, acc = 4009950248756219
# seed = 144, shot = 4, acc = 48855721393034823
# seed = 144, shot = 8, acc = 518905472636816
# seed = 144, shot = 16, acc = 5303482587064676
# seed = 145, shot = 1, acc = 4208955223880597
# seed = 145, shot = 4, acc = 5124378109452736
# seed = 145, shot = 8, acc = 4925373134328358
# seed = 145, shot = 16, acc = 5487562189054727

####################################### tnews ##############################################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                64: 5691542288557214
shot1 = torch.Tensor([38.70646766169154, 40.09950248756219, 42.08955223880597])
shot4 = torch.Tensor([49.651741293532337, 48.855721393034823, 51.24378109452736])
shot8 = torch.Tensor([53.03482587064676, 51.8905472636816, 49.25373134328358])
shot16 = torch.Tensor([55.22388059701493, 53.03482587064676, 54.87562189054727])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))

####################################### cnews ##############################################
# cnews
# seed = 2, shot = 1, acc = 6316
# seed = 2, shot = 4, acc = 7174
# seed = 2, shot = 8, acc = 8198
# seed = 2, shot = 16, acc = 8278
# seed = 143, shot = 1, acc = 6015
# seed = 143, shot = 4, acc = 778
# seed = 143, shot = 8, acc = 8138
# seed = 143, shot = 16, acc = 8368
# seed = 144, shot = 1, acc = 6592
# seed = 144, shot = 4, acc = 7459
# seed = 144, shot = 8, acc = 7758
# seed = 144, shot = 16, acc = 8022

shot1 = torch.Tensor([63.16, 60.15, 65.92])
shot4 = torch.Tensor([71.74, 77.8, 74.59])
shot8 = torch.Tensor([81.98, 81.38, 77.58])
shot16 = torch.Tensor([82.78, 83.68, 80.22])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
#######################################csldcp##############################################
# csldcp
# seed = 2, shot = 1, acc = 38228699551569506
# seed = 2, shot = 4, acc = 43385650224215244
# seed = 2, shot = 8, acc = 4624439461883408
# seed = 2, shot = 16, acc = 5
# seed = 144, shot = 1, acc = 38621076233183854
# seed = 144, shot = 4, acc = 4405829596412556
# seed = 144, shot = 8, acc = 4669282511210762
# seed = 144, shot = 16, acc = 48374439461883406
# seed = 145, shot = 1, acc = 3783632286995516
# seed = 145, shot = 4, acc = 4545964125560538
# seed = 145, shot = 8, acc = 47533632286995514
# seed = 145, shot = 16, acc = 49831838565022424

shot1 = torch.Tensor([38.228699551569506, 38.621076233183854, 37.83632286995516])
shot4 = torch.Tensor([43.385650224215244, 44.05829596412556, 45.45964125560538])
shot8 = torch.Tensor([46.24439461883408,  46.69282511210762, 47.533632286995514])
shot16 = torch.Tensor([50, 48.374439461883406, 49.831838565022424])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
