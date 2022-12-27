import network
d=network.parse_args()

'''Communication Efficiency Exp'''
for d['mode'] in ['ROBUST']:
        d['eta_x'] = 1.
        for d['mu'] in [0.01]:
            for d['scheme'] in ['Sparsification','Quantization']:
                d['gamma']=0.8
                d['save']=0.5
                d['bits']=4
                network.network(d)

'''Regularization Effect Exp'''
d['gamma'] = 1.
d['scheme']='Sparsification'
d['save']=1.
for d['mode'] in ['ROBUST']:
        d['eta_x'] = 1.
        for d['mu'] in [10.,1.,0.01]:
            network.network(d)

for d['mode'] in ['NOT_ROBUST']:
        d['eta_x'] = 2.
        for d['mu'] in [1.]:
            d['gamma']=1.
            d['save']=1.
            d['scheme']='Sparsification'
            network.network(d)

'''Choco-SGD Runs'''
for d['mode'] in ['NOT_ROBUST']:
        d['eta_x'] = 2.
        for d['mu'] in [1.]:
            for d['scheme'] in ['Sparsification','Quantization']:
                d['gamma'] = 0.8
                d['save']=0.5
                d['bits']=4
                network.network(d)


