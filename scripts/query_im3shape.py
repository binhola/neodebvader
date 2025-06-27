from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert
import pandas as pd

print("Querying DES SVA1 im3shape catalogue...")

sql_query = """
SELECT 
    i.coadd_objects_id, 
    i.ra_shift,
    i.dec_shift,
    i.e_1, 
    i.e_2, 
    i.snr_w,
    c.ra_j2000, 
    c.dec_j2000
FROM des_sva1.gold_im3shape AS i
JOIN des_sva1.gold_catalog AS c
    ON i.coadd_objects_id = c.coadd_objects_id
WHERE
    i.e_1 IS NOT NULL AND 
    i.e_2 IS NOT NULL AND 
    i.e_1 != -9999 AND 
    i.e_2 != -9999 AND
    i.snr_w > 20
LIMIT 100000
"""

try:
    # Replace 'your_access_token' with your Data Lab token
    result = qc.query(sql=sql_query, fmt='csv', token='your_access_token')
    df = convert(result, outfmt='pandas')
    if df.empty:
        print("Query returned no results. Check table/columns or token.")
        exit()
    df['ra'] = df['ra_j2000'] + ( df['ra_shift'] / 3600. ) # correction for the position with gold catalogue
    df['dec'] = df['dec_j2000'] + ( df['dec_shift'] / 3600. )
    print("Query succeeded")
except Exception as e:
    print(f"Query failed: {e}")
    print("Verify token at https://datalab.noirlab.edu/account or table 'des_dr1.im3shape'")
    exit()

# Save catalogue
catalogue_file = '../des_sva1_im3shape_ellipticity.csv'
df.to_csv(catalogue_file, index=False)
print(f"Catalogue saved to {catalogue_file}")