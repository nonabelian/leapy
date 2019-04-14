from marshmallow import Schema
from marshmallow import fields


class ModelSchema(Schema):
    class Meta:
        ordered = True
    test_int = fields.Integer()
    test_str = fields.String()

MODEL_SCHEMA = ModelSchema()

FEATURES = list(MODEL_SCHEMA.declared_fields.keys())
