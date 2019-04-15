from marshmallow import Schema
from marshmallow import fields


class ModelSchema(Schema):
    class Meta:
        ordered = True
    dt = fields.String()
    cat_1 = fields.String()
    cat_2 = fields.String()

MODEL_SCHEMA = ModelSchema()

FEATURES = list(MODEL_SCHEMA.declared_fields.keys())
