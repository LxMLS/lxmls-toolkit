from lxmls.sequences.id_feature import IDFeatures


# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        word = unicode(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if unicode.istitle(word):
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if unicode.isdigit(word):
            # Generate feature name.
            feat_name = "number::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if unicode.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in xrange(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in xrange(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features
