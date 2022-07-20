import logging
logger = logging.getLogger('base')


def create_model(opt,args):
    model = opt['model']

    if model == 'CVPR':
        from .IRN_model import IRNModel as M
    elif model == 'PAMI':
        from .IRNp_model import IRNpModel as M
    elif model == 'ICASSP_NOWAY':
        from .IRNcrop_model import IRNcropModel as M
    elif model == 'ICASSP_RHI':
        from .tianchi_model import IRNrhiModel as M
        # from .IRNrhi_model import IRNrhiModel as M
    elif model == 'CLRNet':
        from .IRNclr_model import IRNclrModel as M
    elif model == 'Qian_rumor':
        from .RumorModel import RumorModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt,args)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
