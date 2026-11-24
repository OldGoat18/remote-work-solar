"""Public-source ingestion unit tests which do not download data."""
from wlpr.io.public_inputs import latlon_to_mesh500, mesh500_centroid


def test_jis_mesh500_round_trip():
    for mesh in ["533945263", "533946111", "523975432", "543900004"]:
        assert latlon_to_mesh500(*mesh500_centroid(mesh)) == mesh


def test_public_sources_are_registered():
    from wlpr.io import paths
    registered = paths._inputs_index()
    expected = {
        "estat_T001194_11", "estat_T001194_12", "estat_T001194_13", "estat_T001194_14",
        "meteostat_hourly_47626", "meteostat_hourly_47662",
        "meteostat_hourly_47670", "meteostat_hourly_47682", "repos_R3_pv_grid500",
    }
    assert expected <= set(registered)
