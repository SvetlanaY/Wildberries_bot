from pymystem3 import Mystem


def pos_define(x):
    m = Mystem()
    w = m.analyze(x)[0]
    pos = w['analysis'][0]['gr'].split(',')[0]
    pos = pos.split('=')[0].strip()
    return pos


def final_topics(df):
    final_topics = []
    for x in df.index:
        if len(final_topics)<5:
            if (pos_define(x) not in ['V','ADV']) and (x not in final_topics):
                
                final_topics.append(x)
        else: break
    return final_topics