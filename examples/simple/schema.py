from marshmallow import Schema
from marshmallow import fields


FEATURES_SCHEMA = [('dt', fields.Str()),
                  ('cat_1', fields.Str()),
                  ('cat_2', fields.Str())]

FEATURES = [f for f, _ in FEATURES_SCHEMA]


class ModelSchema(Schema):
    class Meta:
        ordered = True

for feature, dtype in FEATURES_SCHEMA:
    setattr(ModelSchema, feature, dtype)

model_schema = ModelSchema()
