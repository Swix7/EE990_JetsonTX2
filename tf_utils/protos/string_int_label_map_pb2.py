# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tf_utils/protos/string_int_label_map.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='tf_utils/protos/string_int_label_map.proto',
  package='tf_utils.protos',
  serialized_pb=_b('\n*tf_utils/protos/string_int_label_map.proto\x12\x0ftf_utils.protos\"G\n\x15StringIntLabelMapItem\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\x05\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\"I\n\x11StringIntLabelMap\x12\x34\n\x04item\x18\x01 \x03(\x0b\x32&.tf_utils.protos.StringIntLabelMapItem')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_STRINGINTLABELMAPITEM = _descriptor.Descriptor(
  name='StringIntLabelMapItem',
  full_name='tf_utils.protos.StringIntLabelMapItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tf_utils.protos.StringIntLabelMapItem.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='id', full_name='tf_utils.protos.StringIntLabelMapItem.id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='tf_utils.protos.StringIntLabelMapItem.display_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=134,
)


_STRINGINTLABELMAP = _descriptor.Descriptor(
  name='StringIntLabelMap',
  full_name='tf_utils.protos.StringIntLabelMap',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='item', full_name='tf_utils.protos.StringIntLabelMap.item', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=209,
)

_STRINGINTLABELMAP.fields_by_name['item'].message_type = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMapItem'] = _STRINGINTLABELMAPITEM
DESCRIPTOR.message_types_by_name['StringIntLabelMap'] = _STRINGINTLABELMAP

StringIntLabelMapItem = _reflection.GeneratedProtocolMessageType('StringIntLabelMapItem', (_message.Message,), dict(
  DESCRIPTOR = _STRINGINTLABELMAPITEM,
  __module__ = 'tf_utils.protos.string_int_label_map_pb2'
  # @@protoc_insertion_point(class_scope:tf_utils.protos.StringIntLabelMapItem)
  ))
_sym_db.RegisterMessage(StringIntLabelMapItem)

StringIntLabelMap = _reflection.GeneratedProtocolMessageType('StringIntLabelMap', (_message.Message,), dict(
  DESCRIPTOR = _STRINGINTLABELMAP,
  __module__ = 'tf_utils.protos.string_int_label_map_pb2'
  # @@protoc_insertion_point(class_scope:tf_utils.protos.StringIntLabelMap)
  ))
_sym_db.RegisterMessage(StringIntLabelMap)


# @@protoc_insertion_point(module_scope)
