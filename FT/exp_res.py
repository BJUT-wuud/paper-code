import torch


# tnews
# seed = 2, 1: 23057768924302788, 4: 4437250996015936, 8: 4900398406374502, 16:  50199203187251
# seed = 144, 1:  26095617529880477, 4:3610557768924303, 8: 424800796812749, 16: 4910358565737052
# seed = 145, 1: 2793824701195219, 4: 44721115537848605, 8: 4945219123505976, 16: 5189243027888446


####################################### tnews ##############################################                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                64: 5691542288557214
shot1 = torch.Tensor([23.057768924302788, 26.095617529880477, 27.93824701195219])
shot4 = torch.Tensor([44.37250996015936, 36.10557768924303, 44.721115537848605])
shot8 = torch.Tensor([49.00398406374502, 42.4800796812749, 49.45219123505976])
shot16 = torch.Tensor([50.199203187251, 49.10358565737052, 51.89243027888446])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))

####################################### cnews ##############################################
# cnews
# seed = 2, 1:  5428, 4: 794, 8:9008, 16:9082
# seed = 143, 1: 4165, 4: 7681, 8:8245, 16:9303
# seed = 144, 1: 6787, 4: 891, 8:9154, 16:924

shot1 = torch.Tensor([54.28, 41.65, 67.87])
shot4 = torch.Tensor([79.4, 76.81, 89.1])
shot8 = torch.Tensor([90.08, 82.45, 91.54])
shot16 = torch.Tensor([90.82, 93.03, 92.4])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
#######################################csldcp##############################################
# csldcp
# seed=2, 1: 25, 4: 42096412556053814, 8:5011210762331838, 16: 531390134529148
# seed=144, 1: 2875560538116592, 4:  43385650224215244, 8:4646860986547085, 16: 5179372197309418
# seed=145, 1: 24663677130044842, 4: 4316143497757848, 8:4899103139013453, 16: 5358744394618834

shot1 = torch.Tensor([25, 28.75560538116592, 24.663677130044842])
shot4 = torch.Tensor([42.096412556053814, 43.385650224215244, 43.16143497757848])
shot8 = torch.Tensor([50.11210762331838,  46.46860986547085,48.99103139013453])
shot16 = torch.Tensor([53.1390134529148, 51.79372197309418, 53.58744394618834])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
