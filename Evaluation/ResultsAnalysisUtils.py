import matplotlib.pyplot as plt
import math

linefmt_list = ['--', '-.', ':']
marker_list = ['o', '*', 'D', 'h', 'p', 's', 'v', '^', '<', '>']

def plot_experiments_metric_curves_by_profile_size(
        expkeys, expnames, dframes_manager, metric, max_profsize=99999999,
        last_purchase_only=False, original_only=False, figsize=(8, 4),
        accumulative=False, sort_by_last=False):
    
    from TransactionsUtils import TransactionsHandler
    last_purchase_map = TransactionsHandler.last_purchase_map
    user_profile_map = TransactionsHandler.user_profile_map

    plt.figure(figsize=figsize)
    tmp = []
    for expkey, expname in zip(expkeys, expnames):
        dframes_manager.exec_void_func(expkey, 'set_%s_column' % metric)
        df = dframes_manager.get_dataframe(expkey)
        pairs = []
        for t, cid, met in zip(df.timestamp, df.customer_id, df[metric]):
            key = (t.to_pydatetime(), cid)
            if last_purchase_only and not last_purchase_map[key]:
                continue
            profile = user_profile_map[key]
            profsize = len(profile)
            if profsize > max_profsize:
                continue
            pairs.append((profsize, met))
        pairs.sort()
        sorted_profsizes = [p[0] for p in pairs]
        sorted_mets = [p[1] for p in pairs]
        met_means = []
        met_errors = []
        unique_profsizes = []
        offset = 0
        if accumulative:
            for i, profsize in enumerate(sorted_profsizes):
                if (i+1 == len(pairs) or pairs[i+1][0] > profsize):
                    count = i+1
                    mean = sum(sorted_mets[j] for j in range(0, i+1)) / count
                    std = math.sqrt(sum((sorted_mets[j] - mean)**2 for j in range(0, i+1)) / count)
                    met_means.append(mean)
                    met_errors.append(std / math.sqrt(count))
                    unique_profsizes.append(profsize)
        else:
            for i, profsize in enumerate(sorted_profsizes):
                if (i+1 == len(pairs) or pairs[i+1][0] > profsize):
                    count = i - offset + 1
                    mean = sum(sorted_mets[j] for j in range(offset, i+1)) / count
                    std = math.sqrt(sum((sorted_mets[j] - mean)**2 for j in range(offset, i+1)) / count)
                    met_means.append(mean)
                    met_errors.append(std / math.sqrt(count))
                    unique_profsizes.append(profsize)
                    offset = i+1
        
        tmp.append((met_means[-1], met_means, met_errors, unique_profsizes, expname))
    
    if sort_by_last:
        tmp.sort(reverse=True)
    
    for idx, (_, met_means, met_errors, unique_profsizes, expname) in enumerate(tmp):
        plt.errorbar(unique_profsizes, met_means, met_errors,
            label=expname,
            color='C%d'%idx,
            fmt=linefmt_list[idx % len(linefmt_list)], 
            marker=marker_list[idx % len(marker_list)],
            markersize=5,
            linewidth=1)
    plt.title('%s vs profile size' % metric)
    plt.xlabel('profile size')
    plt.ylabel(metric)
    plt.legend()
    plt.show()