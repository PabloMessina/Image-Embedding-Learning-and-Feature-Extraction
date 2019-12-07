from Config import Config
from Classes import UploadEvent, PurchaseSessionEvent, Artwork

def get_upload_events(df):
    events = [
        UploadEvent(upload_date, artwork_id) for artwork_id, upload_date\
        in zip(df.id, df.upload_date)
    ]
    events.sort(key=lambda e: e.timestamp)
    return events

def get_purchase_session_events(df):
    art_ids_dict = dict()
    
    for customer_id, artwork_id, order_date in zip(
        df.customer_id, df.artwork_id, df.order_date
    ):
        # append artwork_id to corresponding purchase list
        key = (order_date, customer_id)
        try:
            art_ids_dict[key].append(artwork_id)
        except KeyError:
            art_ids_dict[key] = [artwork_id]

    # build event list from dictionary
    events = [PurchaseSessionEvent(order_date, artwork_ids, customer_id)\
                for (order_date, customer_id), artwork_ids in art_ids_dict.items()]
    events.sort(key=lambda e: e.timestamp)
    # return events
    return events

def get_artwork_dict(df, extra_columns=None):
    colnames = ['id', 'upload_date', 'original']
    if extra_columns:
        colnames.extend(extra_columns)
    columns = [df[col] for col in colnames]    
    n_cols = len(colnames)
    artworks_dict = {}
    for args in zip(*columns):
        aid = args[0]
        upload_date = args[1]
        original = args[2] == 't'
        kwargs = { colnames[i] : args[i] for i in range(3, n_cols) }
        artworks_dict[aid] = Artwork(aid, upload_date, original, **kwargs)
    return artworks_dict


class __TransactionsHandler:

    def __init__(self):
        pass    
    
    @property
    def valid_artworks_df(self):
        try:
            df = self._VALID_ARTWORKS_DF
        except AttributeError:
            import pandas as pd
            df = pd.read_csv(Config['VALID_ARTWORKS_FILE'])
            df['upload_date'] = pd.to_datetime(df['upload_date'])
            self._VALID_ARTWORKS_DF = df
        return df

    @property
    def valid_sales_df(self):
        try:
            df = self._VALID_SALES_DF
        except AttributeError:
            import pandas as pd
            df = pd.read_csv(Config['VALID_SALES_FILE'])
            df['order_date'] = pd.to_datetime(df['order_date'])
            self._VALID_SALES_DF = df
        return df

    @property
    def purchase_session_events(self):
        try:
            return self._PURCHASE_SESSION_EVENTS
        except AttributeError:
            self._PURCHASE_SESSION_EVENTS = get_purchase_session_events(self.valid_sales_df)
            return self._PURCHASE_SESSION_EVENTS

    @property
    def upload_events(self):
        try:
            return self._UPLOAD_EVENTS
        except AttributeError:
            self._UPLOAD_EVENTS = get_upload_events(self.valid_artworks_df)
            return self._UPLOAD_EVENTS

    @property
    def artworks_dict(self):
        try:
            return self._ARTWORKS_DICT
        except AttributeError:
            self._ARTWORKS_DICT = get_artwork_dict(self.valid_artworks_df)
            return self._ARTWORKS_DICT
    
    def _init_purchase_session_map(self):
        psmap__set = dict()
        psmap__list = dict()
        psevents = self.purchase_session_events
        for pse in psevents:
            key = (pse.timestamp, pse.customer_id)
            psmap__list[key] = pse.artwork_ids
            psmap__set[key] = set(pse.artwork_ids)
        self._PURCHASE_SESSION_MAP__LIST = psmap__list
        self._PURCHASE_SESSION_MAP__SET = psmap__set
    
    @property
    def purchase_session_map__list(self):
        try:
            return self._PURCHASE_SESSION_MAP__LIST
        except AttributeError:
            self._init_purchase_session_map()
            return self._PURCHASE_SESSION_MAP__LIST
    
    @property
    def purchase_session_map__set(self):
        try:     
            return self._PURCHASE_SESSION_MAP__SET
        except AttributeError:
            self._init_purchase_session_map()
            return self._PURCHASE_SESSION_MAP__SET

    @property
    def user_profile_map(self):
        try:
            return self._USER_PROFILE_MAP
        except AttributeError:
            current_profiles = dict()
            user_profile_map = dict()
            _empty_array = []
            psevents = self.purchase_session_events
            for pse in psevents:
                cid = pse.customer_id        
                key = (pse.timestamp, cid)
                profile_before = current_profiles.get(cid, _empty_array)
                user_profile_map[key] = profile_before
                current_profiles[cid] = profile_before + pse.artwork_ids
            self._USER_PROFILE_MAP = user_profile_map
            return self._USER_PROFILE_MAP
        
    @property
    def relevant_ids_map(self):
        try:
            return self._RELEVANT_IDS_MAP
        except AttributeError:
            rel_ids_map = dict()
            last_rel_ids = dict()
            psevents = self.purchase_session_events
            upevents = self.upload_events
            artworks_dict = self.artworks_dict
            for pse in reversed(psevents): # backwards in time
                cid = pse.customer_id
                try:
                    rel_ids = last_rel_ids[cid]
                except KeyError:
                    rel_ids = last_rel_ids[cid] = set()
                rel_ids.update(pse.artwork_ids)
                key = (pse.timestamp, cid)
                rel_ids_map[key] = rel_ids.copy()
            merged_events = upevents + psevents
            merged_events.sort(key=lambda e : e.timestamp)
            inventory = set()
            for e in merged_events:
                if e.type == 'upload':
                    inventory.add(e.artwork_id)
                else:
                    key = (e.timestamp, e.customer_id)
                    rel_ids_map[key] = (rel_ids_map[key] & inventory)
                    assert len(rel_ids_map[key]) > 0
                    for aid in e.artwork_ids:
                        if artworks_dict[aid].original:
                            inventory.remove(aid)
            self._RELEVANT_IDS_MAP = rel_ids_map
            return self._RELEVANT_IDS_MAP
    
    @property
    def last_purchase_map(self):
        try:
            return self._LAST_PURCHASE_MAP
        except AttributeError:
            psevents = self.purchase_session_events
            encountered_cids = set()
            first_keys = set()
            for pse in psevents:
                cid = pse.customer_id
                key = (pse.timestamp, cid)
                if cid not in encountered_cids:
                    encountered_cids.add(cid)
                    first_keys.add(key)
            encountered_cids.clear()
            last_purchase_map = dict()
            for pse in reversed(psevents):
                cid = pse.customer_id
                key = (pse.timestamp, cid)
                is_last = False
                if cid not in encountered_cids:
                    encountered_cids.add(cid)
                    if key not in first_keys:
                        is_last = True
                last_purchase_map[key] = is_last
            self._LAST_PURCHASE_MAP = last_purchase_map
            return self._LAST_PURCHASE_MAP

# singlenton instance
TransactionsHandler = __TransactionsHandler()