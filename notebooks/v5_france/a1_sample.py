"""A1: write pair samples for adversarial validation: ~400k France test, ~400k India+US test, ~400k train
(fit+tune+holdout, in_group rows, with label) — all v4 FEATURES + country, to artifacts/v5_tmp/adv_*.parquet."""
from common import *
from ber.features import FEATURES
c = connect()
cols = ", ".join(f"f.{x}" for x in FEATURES)
c.execute(f"CREATE TABLE s1c AS SELECT entity_id, country FROM '{CACHE}/raw_test_s1.parquet'")
c.execute(f"""COPY (SELECT f.lid, f.rid, s.country, {cols} FROM '{FEAT['test']}' f JOIN s1c s ON s.entity_id = f.lid
    WHERE (s.country = 'France' AND hash(f.lid || f.rid) % 1000 < 75) OR (s.country <> 'France' AND hash(f.lid || f.rid) % 1000 < 30))
    TO '{TMP}/adv_test.parquet' (FORMAT parquet)""")
c.execute(f"""COPY (SELECT f.lid, f.rid, n.country_norm AS country, f.label, {cols}
    FROM read_parquet(['{FEAT['fit']}', '{FEAT['tune']}', '{FEAT['holdout']}']) f
    JOIN '{CACHE}/norm_train_s1.parquet' n ON n.entity_id = f.lid
    WHERE f.in_group = 1 AND hash(f.lid || f.rid) % 1000 < 30) TO '{TMP}/adv_train.parquet' (FORMAT parquet)""")
print(c.execute(f"SELECT country, count(*) FROM '{TMP}/adv_test.parquet' GROUP BY 1").fetchall())
print(c.execute(f"SELECT country, count(*), avg(label) FROM '{TMP}/adv_train.parquet' GROUP BY 1").fetchall())
