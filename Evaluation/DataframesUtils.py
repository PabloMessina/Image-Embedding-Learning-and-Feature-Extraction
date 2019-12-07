import os
from itertools import repeat as iter_repeat
import pandas as pd
from Config import Config
from MetricsUtils import precision, recall, nDCG, average_precision, auc_exact, reciprocal_rank
from TransactionsUtils import TransactionsHandler as _TransactionsHandler

def _almost_equals(x, y):
    return abs(x - y) < 1e-3

class _DataframeWrapper:
    def __init__(self, expkey, conn):
        self.conn = conn
        self.expkey = expkey
        self.filepath = os.path.join(Config['EXPERIMENT_RESULTS_DIRPATH'], expkey)
        self.df = None
        self.modiftime = None
        self.table_row_id = None
        self.metrics_memo = dict()
        self.values_memo = dict()
        self.exec_memo = set()
    
    def get_dataframe(self):
        df = self.df
        if df is None:
            path = self.filepath
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.df = df
        return df
    
    def exec_void_func(self, func_name, dfmanager):
        exec_memo = self.exec_memo
        if func_name not in exec_memo:
            df = self.get_dataframe()
            func = dfmanager.void_funcs[func_name]
            func(df, self.expkey, dfmanager)
            exec_memo.add(func_name)            

    def load_cached_metrics_from_DB(self):
        if self.table_row_id is not None:
            return
        assert len(self.metrics_memo) == 0
        conn = self.conn
        row = conn.execute(
            'SELECT * FROM metrics WHERE expkey = \'%s\'' % self.expkey).first()        
        self.modiftime = os.stat(self.filepath).st_mtime
        if row is None:
            row = conn.execute('''INSERT INTO metrics (expkey, modiftime)
            VALUES (\'%s\', %.10f) RETURNING id''' % (self.expkey, self.modiftime)).first()
            self.table_row_id = row['id']
        else:
            self.table_row_id = row['id']            
            prev_modiftime = row['modiftime']

            # assertions
            assert prev_modiftime is not None
            try:
                assert self.modiftime > prev_modiftime or\
                _almost_equals(self.modiftime, prev_modiftime)
            except AssertionError:
                print('self.modiftime = ', self.modiftime)
                print('prev_modiftime = ', prev_modiftime)
                print('expkey = ', self.expkey)
                raise

            metric_cols_names = set(row.keys()) - {'id', 'expkey', 'modiftime'}
            if _almost_equals(self.modiftime, prev_modiftime):
                for key in metric_cols_names:
                    val = row[key]
                    if val is not None:
                        self.metrics_memo[key] = val
            else:
                print('** outdated modiftime in db detected, prev=%.10f, curr=%.10f, expkey = %s'\
                % (prev_modiftime, self.modiftime, self.expkey))
                updates = ['modiftime = %.10f' % self.modiftime]
                updates.extend(('%s = NULL' % x) for x in metric_cols_names)
                updates_string = ','.join(updates)
                self.conn.execute(
                    'UPDATE metrics SET %s WHERE id=%d' % (updates_string, self.table_row_id))
    
    def get_metric(self, metric_name, dfmanager):
        metrics_memo = self.metrics_memo
        try:
            return metrics_memo[metric_name]
        except KeyError:
            if self.table_row_id is None:
                self.load_cached_metrics_from_DB()
                assert self.table_row_id is not None
                try:
                    return metrics_memo[metric_name]
                except KeyError:
                    pass
            df = self.get_dataframe()
            func = dfmanager.metric_funcs[metric_name]
            val = func(df, self.expkey, dfmanager)
            metrics_memo[metric_name] = val
            self.conn.execute('UPDATE metrics SET %s = %.10f WHERE id=%d' % (
                metric_name, val, self.table_row_id))
            return val

    def get_value(self, value_name, dfmanager):
        values_memo = self.values_memo
        try:
            return values_memo[value_name]
        except KeyError:
            df = self.get_dataframe()
            func = dfmanager.value_funcs[value_name]
            val = func(df, self.expkey, dfmanager)
            values_memo[value_name] = val
            return val

class _DataframesMetricsManager:

    _global_metricsdb_conn = None
    _global_metricsdb_engine = None
    _global_transactions_handler = None
    _global_metadata_cache = None
    _global_jaccard_sim_cache = None
    _num_users = None
    _num_purch_sess = None

    @classmethod
    def _init_global_metricsdb_conn_engine(cls):
        if cls._global_metricsdb_conn is None:            
            from sqlalchemy import create_engine
            engine = create_engine(Config['SQLALCHEMY_METRICSDB_STRING'])
            cls._global_metricsdb_engine = engine
            cls._global_metricsdb_conn = engine.connect()
            print('cls._global_metricsdb_engine initialized')
            print('cls._global_metricsdb_conn initialized')

    @classmethod
    def _get_global_metricsdb_engine(cls):
        cls._init_global_metricsdb_conn_engine()
        return cls._global_metricsdb_conn

    @classmethod
    def _get_global_metricsdb_conn(cls):
        cls._init_global_metricsdb_conn_engine()
        return cls._global_metricsdb_conn

    @classmethod
    def get_global_metadata_cache(cls):
        if cls._global_metadata_cache is None:
            from .FeaturesUtils import MetadataCache
            cls._global_metadata_cache = MetadataCache
        return cls._global_metadata_cache

    @classmethod
    def get_global_jaccard_sim_cache(cls):
        if cls._global_jaccard_sim_cache is None:
            from .FeaturesUtils import JaccardSimilarityCache
            metadata_cache = cls.get_global_metadata_cache()
            offset = len(metadata_cache.get_all_attr_values())
            def metadata_ids_getter(art_id):
                ids = set()
                ids.update(metadata_cache.get_color_value_ids(art_id))
                ids.update(metadata_cache.get_medium_value_ids(art_id))
                ids.add(offset + metadata_cache.get_artist_id(art_id))
                return ids
            cls._global_jaccard_sim_cache = JaccardSimilarityCache(metadata_ids_getter)
        return cls._global_jaccard_sim_cache

    @classmethod
    def purchase_sessions_count(cls):
        if cls._num_purch_sess is not None:
            return cls._num_purch_sess
        cls._init_global_metricsdb_conn_engine()
        from .DBUtils import count_purchase_sessions
        cls._num_purch_sess = count_purchase_sessions(cls._global_metricsdb_conn, original_only=True)
        return cls._num_purch_sess
    
    @classmethod
    def users_count(cls):
        if cls._num_users is not None:
            return cls._num_users
        cls._init_global_metricsdb_conn_engine()
        from .DBUtils import count_users
        cls._num_users = count_users(cls._global_metricsdb_conn, original_only=True)
        return cls._num_users

    def __init__(self):
        self.dfwrappers = dict()
        self.void_funcs = dict()
        self.metric_funcs = dict()
        self.value_funcs = dict()
        self.engine = self._get_global_metricsdb_engine()
        self.conn = self._get_global_metricsdb_conn()
        self._init_metrics_table()

    def _init_metrics_table(self):
        conn = self.conn
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS metrics (
                id          SERIAL          PRIMARY KEY,
                expkey      VARCHAR(255)    NOT NULL,
                modiftime   FLOAT           NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS expkey_index ON metrics(expkey);
            ''')

    def set_metric_func(self, metric_name, func, k=None, db_column_type='DECIMAL', override=False):
        actual_func = func if k is None else lambda *args: func(*args, k)
        self.metric_funcs[metric_name] = actual_func
        conn = self.conn
        if override:
            conn.execute('ALTER TABLE metrics DROP COLUMN IF EXISTS %s;' % metric_name)
            for dfw in self.dfwrappers.values():
                dfw.metrics_memo.pop(metric_name, None)
                assert metric_name not in dfw.metrics_memo
        conn.execute(
            'ALTER TABLE metrics ADD COLUMN IF NOT EXISTS %s %s;' % (metric_name, db_column_type))
    
    def set_value_func(self, val_name, func, override=False):
        self.value_funcs[val_name] = func
        if override:
            for dfw in self.dfwrappers.values():
                dfw.values_memo.pop(val_name, None)
                assert val_name not in dfw.values_memo
        
    def set_void_func(self, func_name, func, k=None, override=False):
        actual_func = func if k is None else lambda *args: func(*args, k)
        self.void_funcs[func_name] = actual_func
        if override:
            for dfw in self.dfwrappers.values():
                dfw.exec_memo.discard(func_name)
                assert func_name not in dfw.exec_memo

    
    def _get_dfwrapper(self, expkey):
        try:
            dfw = self.dfwrappers[expkey]
        except KeyError:
            dfw = _DataframeWrapper(expkey, self.conn)
            self.dfwrappers[expkey] = dfw
        return dfw
    
    def get_dataframe(self, expkey):
        dfw = self._get_dfwrapper(expkey)
        return dfw.get_dataframe()

    def exec_void_func(self, expkey, func_name):
        dfw = self._get_dfwrapper(expkey)
        dfw.exec_void_func(func_name, self)

    def _minibatch_exec_metric_func(self, dfws, metric_name):
        thread_conn = self.engine.connect()
        for dfw in dfws:
            tmp = dfw.conn
            dfw.conn = thread_conn
            dfw.get_metric(metric_name, self)
            dfw.conn = tmp
        thread_conn.close()

    def batch_exec_metric_func(self, expkeys, metric_name):
        assert len(expkeys) > 0
        from multiprocessing.dummy import Pool as ThreadPool
        dfws = []
        for key in expkeys:
            dfw = self._get_dfwrapper(key)
            dfw.load_cached_metrics_from_DB()
            if metric_name not in dfw.metrics_memo:
                dfws.append(dfw)

        n = len(dfws)
        if n < 80:
            for dfw in dfws:
                dfw.get_metric(metric_name, self)
        else:
            print('--- batch_exec_metric_func()')
            print('len(dfws) = %d -> using thread pool' % len(dfws))
            N_THREADS = 4
            minibatchsize = n // N_THREADS + (n % N_THREADS > 0)
            pool = ThreadPool(N_THREADS)
            pool.starmap(
                self._minibatch_exec_metric_func,
                zip((dfws[i * minibatchsize : (i+1) * minibatchsize] for i in range(N_THREADS)),
                    iter_repeat(metric_name)))
            pool.close() 
            pool.join()
            print('All %d threads successfully executed' % N_THREADS)
        
        # last sanity check assert
        for dfw in dfws:
            assert metric_name in dfw.metrics_memo

    def get_metric(self, expkey, metric_name):
        dfw = self._get_dfwrapper(expkey)
        return dfw.get_metric(metric_name, self)
    
    def get_value(self, expkey, value_name):
        dfw = self._get_dfwrapper(expkey)
        return dfw.get_value(value_name, self)
    
    # ========= STANDARD FUNCTIONS ========    

    @staticmethod
    def get_customer_groupby(df, key, df_manager):
        return df.groupby('customer_id')

    @staticmethod
    def get_user_coverage(df, key, df_manager):
        gb = df_manager.get_value(key, 'customer_gb')
        return len(gb) / _DataframesMetricsManager.users_count()
    
    @staticmethod
    def get_session_coverage(df, key, df_manager):
        return len(df) / _DataframesMetricsManager.purchase_sessions_count()

    @staticmethod
    def set_rec_ids_list_column(df, key, df_manager):
        rec_ids_lists = []
        for _str_ids in df.recommended_ids:
            str_ids = str(_str_ids)
            if str_ids != 'nan':
                rec_ids_lists.append(list(map(int, str_ids.split('|'))))
            else:
                rec_ids_lists.append([])
        df['rec_ids_list'] = pd.Series(rec_ids_lists, index=df.index)
    
    @staticmethod
    def set_gt_idxs_list_column(df, key, df_manager):
        try:
            gt_idxs_lists = []
            for _str_ids in df.ground_truth_indexes:
                str_ids = str(_str_ids)
                if str_ids != 'nan':
                    gt_idxs_lists.append(list(map(int, str_ids.split('|'))))
                else:
                    gt_idxs_lists.append([])
            df['gt_idxs_list'] = pd.Series(gt_idxs_lists, index=df.index)
        except AttributeError:
            print(key)
            raise
    
    @staticmethod
    def set_last_purchase_column(df, key, df_manager):
        last_purchase_map = _TransactionsHandler.last_purchase_map
        values = [last_purchase_map[(t.to_pydatetime(), cid)]
            for cid, t in zip(df.customer_id, df.timestamp)]
        df['last_purchase'] = pd.Series(values, index=df.index)

    @staticmethod
    def set_profile_size_column(df, key, df_manager):
        user_profile_map = _TransactionsHandler.user_profile_map
        values = [len(user_profile_map[(t.to_pydatetime(), cid)])
            for cid, t in zip(df.customer_id, df.timestamp)]
        df['profile_size'] = pd.Series(values, index=df.index)

    @staticmethod
    def set_prec_column(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        precs = [precision(gt_idxs, k) for gt_idxs in df.gt_idxs_list]
        df['prec_at%d' % k] = pd.Series(precs, index=df.index)

    @staticmethod
    def get_precision(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_prec_at%d_column' % k)
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['prec_at%d' % k].mean().mean()
    
    @staticmethod
    def get_precision__last_purchase(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_prec_at%d_column' % k)
        return df['prec_at%d' % k][df.last_purchase].mean()

    @staticmethod
    def set_rec_column(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        recs = [recall(gt_idxs, k) for gt_idxs in df.gt_idxs_list]
        df['rec_at%d' % k] = pd.Series(recs, index=df.index)

    @staticmethod
    def get_recall(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_rec_at%d_column' % k)
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['rec_at%d' % k].mean().mean()

    @staticmethod
    def get_recall__last_purchase(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_rec_at%d_column' % k)
        return df['rec_at%d' % k][df.last_purchase].mean()

    @staticmethod
    def set_f1_column(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_prec_at%d_column' % k)
        df_manager.exec_void_func(key, 'set_rec_at%d_column' % k)
        f1s = [2*p*r / (p+r) if (p+r) > 0.0 else 0.0
                for p, r in zip(df['prec_at%d' % k], df['rec_at%d' % k])]
        df['f1_at%d' % k] = pd.Series(f1s, index=df.index)

    @staticmethod
    def get_f1score(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_f1_at%d_column' % k)
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['f1_at%d' % k].mean().mean()

    @staticmethod
    def get_f1score__last_purchase(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_f1_at%d_column' % k)
        return df['f1_at%d' % k][df.last_purchase].mean()

    @staticmethod
    def set_ndcg_column(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        ndcgs = [nDCG(gt_idxs, k) for gt_idxs in df.gt_idxs_list]
        df['ndcg_at%d' % k] = pd.Series(ndcgs, index=df.index)

    @staticmethod
    def get_ndcg(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_ndcg_at%d_column' % k)
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['ndcg_at%d' % k].mean().mean()
    
    @staticmethod
    def get_ndcg__last_purchase(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_ndcg_at%d_column' % k)
        return df['ndcg_at%d' % k][df.last_purchase].mean()

    @staticmethod
    def set_ap_column(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        aps = [average_precision(gt_idxs, k) for gt_idxs in df.gt_idxs_list]
        df['ap_at%d' % k] = pd.Series(aps, index=df.index)

    @staticmethod
    def get_map(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_ap_at%d_column' % k)
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['ap_at%d' % k].mean().mean()

    @staticmethod
    def get_map__last_purchase(df, key, df_manager, k):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_ap_at%d_column' % k)
        return df['ap_at%d' % k][df.last_purchase].mean()

    @staticmethod
    def set_auc_column(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        aucs = [auc_exact(gt_idxs_list, inv_size) for
            gt_idxs_list, inv_size in zip(df.gt_idxs_list, df.inventory_size)]
        df['auc'] = pd.Series(aucs, index=df.index) 
    
    @staticmethod
    def get_auc(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_auc_column')
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['auc'].mean().mean()
    
    @staticmethod
    def get_auc__last_purchase(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_auc_column')
        return df.auc[df.last_purchase].mean()

    @staticmethod
    def set_rr_column(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_gt_idxs_list_column')
        rrs = [reciprocal_rank(gt_idxs_list[0]) for gt_idxs_list in df.gt_idxs_list]
        df['rr'] = pd.Series(rrs, index=df.index)

    @staticmethod
    def get_mrr(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_rr_column')
        gb = df_manager.get_value(key, 'customer_gb')
        return gb['rr'].mean().mean()

    @staticmethod
    def get_mrr__last_purchase(df, key, df_manager):
        df_manager.exec_void_func(key, 'set_last_purchase_column')
        df_manager.exec_void_func(key, 'set_rr_column')
        return df.rr[df.last_purchase].mean()

    @staticmethod
    def set_artist_sharing_column(df, key, df_manager):
        artist_sharing_map = _TransactionsHandler.artist_sharing_map
        values = [artist_sharing_map[(t.to_pydatetime(), cid)]
            for cid, t in zip(df.customer_id, df.timestamp)]
        df['artist_sharing'] = pd.Series(values, index=df.index)


DataframesMetricsManager = _DataframesMetricsManager()