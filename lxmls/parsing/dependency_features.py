import codecs

import numpy as np


class DependencyFeatures:
    """
    Dependency features class
    """

    def __init__(self, use_lexical=False, use_distance=False, use_contextual=False):
        self.feat_dict = {}
        self.n_feats = 0
        self.use_lexical = use_lexical
        self.use_distance = use_distance
        self.use_contextual = use_contextual

    def create_dictionary(self, instances):
        """Creates dictionary of features (note: only uses supported features)"""
        self.feat_dict = {}
        self.n_feats = 0
        for instance in instances:
            nw = np.size(instance.words) - 1
            heads = instance.heads
            for m in range(1, nw+1):
                h = heads[m]
                self.create_arc_features(instance, h, m, True)

        print("Number of features: {0}".format(self.n_feats))

    def create_features(self, instance):
        """Creates arc features from an instance."""
        nw = np.size(instance.words) - 1
        feats = np.empty((nw+1, nw+1), dtype=object)
        for h in range(0, nw+1):
            for m in range(1, nw+1):
                if h == m:
                    feats[h][m] = []
                    continue
                feats[h][m] = self.create_arc_features(instance, h, m)

        return feats

    def create_arc_features(self, instance, h, m, add=False):
        """Creates features for arc h-->m."""
        nw = np.size(instance.words)
        k = 0
        ff = []
        if h < m:
            att_dir = 1  # Right attachment
            dist = m - h  # Distance
        else:
            att_dir = 0  # Left attachment
            dist = h - m  # Distance
        if dist > 10:
            dist = 10
        elif dist > 5:
            dist = 5

        if h == 0:
            hpp = "__START__"
        else:
            hpp = instance.pos[h-1]
        if h == nw-1:
            hpn = "__END__"
        else:
            hpn = instance.pos[h+1]
        if m == 0:
            mpp = "__START__"
        else:
            mpp = instance.pos[m-1]
        if m == nw-1:
            mpn = "__END__"
        else:
            mpn = instance.pos[m+1]

        # Head pos, modifier pos
        f = self.lookup_fid("{0}_{1}_{2}".format(k, instance.pos[h], instance.pos[m]), add)
        ff.append(f)
        k += 1
        # Head pos
        f = self.lookup_fid("{0}_{1}".format(k, instance.pos[h]), add)
        ff.append(f)
        k += 1

        if self.use_lexical:
            # Head word+pos, modifier word+pos
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}".format(
                    k, instance.words[h],
                    instance.pos[h],
                    instance.words[m],
                    instance.pos[m]),
                add)
            ff.append(f)
            k += 1
            # Head word+pos, modifier pos
            f = self.lookup_fid("{0}_{1}_{2}_{3}".format(k, instance.words[h], instance.pos[h], instance.pos[m]), add)
            ff.append(f)
            k += 1
            # Head pos, modifier word
            f = self.lookup_fid("{0}_{1}_{2}_{3}".format(k, instance.pos[h], instance.words[m], instance.pos[m]), add)
            ff.append(f)
            k += 1
            # Head word+pos
            f = self.lookup_fid("{0}_{1}_{2}".format(k, instance.words[h], instance.pos[h]), add)
            ff.append(f)
            k += 1

        if self.use_distance:
            # Direction of attachment
            f = self.lookup_fid("{0}_{1}".format(k, att_dir), add)
            ff.append(f)
            k += 1

            # Distance
            f = self.lookup_fid("{0}_{1}".format(k, dist), add)
            ff.append(f)
            k += 1

        if self.use_contextual:
            # Contextual features
            # Head pos+posl, modifier pos, dir
            f = self.lookup_fid("{0}_{1}_{2}_{3}_{4}".format(k, instance.pos[h], hpp, instance.pos[m], att_dir), add)
            ff.append(f)
            k += 1
            # Head pos+posr, modifier pos, dir
            f = self.lookup_fid("{0}_{1}_{2}_{3}_{4}".format(k, instance.pos[h], hpn, instance.pos[m], att_dir), add)
            ff.append(f)
            k += 1
            # Head pos+posl+posr, modifier pos, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    hpn,
                    instance.pos[m],
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos, modifier pos+posl, dir
            f = self.lookup_fid("{0}_{1}_{2}_{3}_{4}".format(k, instance.pos[h], instance.pos[m], mpp, att_dir), add)
            ff.append(f)
            k += 1
            # Head pos, modifier pos+posr, dir
            f = self.lookup_fid("{0}_{1}_{2}_{3}_{4}".format(k, instance.pos[h], instance.pos[m], mpn, att_dir), add)
            ff.append(f)
            k += 1
            # Head pos, modifier pos+posl+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    instance.pos[m],
                    mpp,
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl, modifier pos+posl, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    instance.pos[m],
                    mpp,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl, modifier pos+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    instance.pos[m],
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posr, modifier pos+posl, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    hpn,
                    instance.pos[m],
                    mpp,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posr, modifier pos+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}".format(
                    k,
                    instance.pos[h],
                    hpn,
                    instance.pos[m],
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl, modifier pos+posl+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    instance.pos[m],
                    mpp,
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posr, modifier pos+posl+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
                    k,
                    instance.pos[h],
                    hpn,
                    instance.pos[m],
                    mpp,
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl+posr, modifier pos+posl, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    hpn,
                    instance.pos[m],
                    mpp,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl+posr, modifier pos+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
                    k,
                    instance.pos[h],
                    hpp,
                    hpn,
                    instance.pos[m],
                    mpn,
                    att_dir),
                add)
            ff.append(f)
            k += 1
            # Head pos+posl+posr, modifier pos+posl+posr, dir
            f = self.lookup_fid(
                "{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}".format(
                    k, instance.pos[h],
                    hpp, hpn, instance.pos[m],
                    mpp, mpn, att_dir),
                add)
            ff.append(f)
            k += 1

        return ff

    def lookup_fid(self, fname, add=False):
        """Looks up dictionary for feature ID."""
        if fname not in self.feat_dict:
            if add:
                fid = self.n_feats
                self.n_feats += 1
                self.feat_dict[fname] = fid
                return fid
            else:
                return -1
        else:
            return self.feat_dict[fname]

    def compute_scores(self, feats, weights):
        """Compute scores by taking the dot product between the feature and weight vector."""
        nw = np.size(feats, 0) - 1
        scores = np.zeros((nw+1, nw+1))
        for h in range(nw+1):
            for m in range(nw+1):
                if feats[h][m] is None:
                    continue
                for f in feats[h][m]:
                    if f < 0:
                        continue
                    scores[h][m] += weights[f]
        return scores
