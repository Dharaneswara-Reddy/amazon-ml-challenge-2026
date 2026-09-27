"""Combine per-country vocab feature files: vf_{fit,tune}_SELF2 = India rows INS2 vocab + US rows USS2 vocab."""
from common import S1, TMP, connect

con = connect("1GB")
con.execute(f"CREATE TABLE ctry AS SELECT entity_id, country_norm c FROM read_parquet('{S1}')")
con.execute(f"""COPY (SELECT v.* FROM read_parquet('{TMP}/vf_tune_INS2.parquet') v JOIN ctry ON entity_id = v.lid WHERE c = 'india'
             UNION ALL SELECT v.* FROM read_parquet('{TMP}/vf_tune_USS2.parquet') v JOIN ctry ON entity_id = v.lid WHERE c = 'us')
             TO '{TMP}/vf_tune_SELF2.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT * FROM read_parquet('{TMP}/vf_fit_INS2_B.parquet') UNION ALL
             SELECT * FROM read_parquet('{TMP}/vf_fit_USUSS2.parquet')) TO '{TMP}/vf_fit_SELF2.parquet' (FORMAT parquet)""")
print("combined")
