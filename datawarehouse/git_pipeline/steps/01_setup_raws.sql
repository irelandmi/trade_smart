USE DATABASE {{environment}};

CREATE OR ALTER TABLE {{environment}}.RAW.TEST_TABLE (
  ID STRING,
  TEXT_FIELD STRING,
  TEXT_FIELD_2 STRING
);

CREATE OR ALTER TABLE {{environment}}.CORE.TEST_TABLE (
  ID STRING,
  TEXT_FIELD STRING,
  TEXT_FIELD_2 STRING
);