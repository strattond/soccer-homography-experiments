from datetime import datetime

import duckdb
import pytest
from squadi_data.fixed import DivisionData, Fixture, FixtureWrapper, Player
from squadi_data.shared import Division

from soccer_homography.db import (
  Match,
  Person,
  PersonParticipationDB,
  addGenericPeopleToMatch,
  ensureGenericPersons,
  getMatchByID,
  importSquadiDivision,
  parseMatchDate,
  upsertMatch,
  upsertPerson,
  upsertPersonParticipation,
)
from soccer_homography.db.persist import transaction

fixedTZ = datetime.now().astimezone().tzinfo


@pytest.mark.usefixtures( "clear_test_database" )
def test_transaction_context_commits_and_rolls_back( conn ):
  with pytest.raises( RuntimeError, match="rollback" ), transaction( conn ):
    conn.execute( "INSERT INTO Person(first_name, last_name) VALUES ('Rolled', 'Back')" )
    raise RuntimeError( "rollback" )

  with transaction( conn ):
    conn.execute( "INSERT INTO Person(first_name, last_name) VALUES ('Committed', 'Person')" )

  assert conn.execute( "SELECT first_name FROM Person" ).fetchall() == [ ( "Committed",) ]


@pytest.mark.usefixtures( "clear_test_database" )
def test_match_squadi_id_is_nullable_and_unique( conn ):
  conn.execute( "INSERT INTO matches(date, home, away, division) VALUES ('2026-09-01', 'A', 'B', 'D')" )
  conn.execute( "INSERT INTO matches(date, home, away, division) VALUES ('2026-09-02', 'C', 'D', 'D')" )
  conn.execute( "INSERT INTO matches(date, home, away, division, squadi_id) VALUES ('2026-09-03', 'E', 'F', 'D', 123)" )

  with pytest.raises( duckdb.ConstraintException ):
    conn.execute( "INSERT INTO matches(date, home, away, division, squadi_id) VALUES ('2026-09-04', 'G', 'H', 'D', 123)" )
  assert conn.execute( "SELECT data_type FROM duckdb_columns() WHERE table_name = 'matches' AND column_name = 'date'" ).fetchone() == ( "TIMESTAMP",)


def test_parse_match_date_as_timestamp():
  assert parseMatchDate( "202609291423" ) == datetime( 2026, 9, 29, 14, 23, tzinfo=fixedTZ )
  assert parseMatchDate( "2026-09-29 14:23" ) == datetime( 2026, 9, 29, 14, 23, tzinfo=fixedTZ )
  assert parseMatchDate( None ) is None
  with pytest.raises( ValueError, match="without a timezone" ):
    parseMatchDate( "2026-09-29T14:23:00+10:00" )


@pytest.mark.usefixtures( "clear_test_database" )
def test_updating_match_without_squadi_id_keeps_existing_squadi_id( conn ):
  match = upsertMatch( conn, Match( id=0, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D", squadi_id=123 ) )

  upsertMatch( conn, Match( id=match.id, date=datetime( 2026, 9, 2, tzinfo=fixedTZ ), home="C", away="D", division="D" ) )

  updated_match = getMatchByID( conn, match.id )
  assert updated_match is not None
  assert updated_match.squadi_id == 123


@pytest.mark.usefixtures( "clear_test_database" )
def test_import_squadi_division_creates_matches_people_and_participations( conn ):
  existing_person_id = conn.execute( "INSERT INTO Person(first_name, last_name) VALUES ('Same', 'Person') RETURNING id" ).fetchone()[ 0 ]
  division = DivisionData(
      div=Division( name="Premier", divisionId=10, teamId=20 ),
      matches=[
          FixtureWrapper(
              match=Fixture(
                  id=100,
                  date="202609011230",
                  players=[ Player( shirt=7, name="Same Person", goals=0, yellows=0, reds=0 ),
                            Player( shirt=12, name="Jane Doe Smith", goals=0, yellows=0, reds=0 ) ],
              )
          ),
          FixtureWrapper(
              match=Fixture(
                  id=101,
                  date="202609021230",
                  players=[ Player( shirt=7, name="Same Person", goals=0, yellows=0, reds=0 ),
                            Player( shirt=4, name="New Person", goals=0, yellows=0, reds=0 ) ],
              )
          ),
      ],
  )

  result = importSquadiDivision( conn, division, { 100: ( "Home United", "Away Rovers" )} )
  repeated_import = importSquadiDivision( conn, division )

  assert result.matches_created == 2
  assert result.persons_created == 2
  assert result.participations_created == 4
  assert repeated_import.matches_created == 0
  assert repeated_import.persons_created == 0
  assert repeated_import.participations_created == 0
  assert repeated_import.participations_updated == 4
  assert conn.execute( "SELECT id, date, home, away, division, squadi_id FROM matches ORDER BY squadi_id" ).fetchall() == [
      ( 1, datetime( 2026, 9, 1, 12, 30 ), "Home United", "Away Rovers", "Premier", 100 ),
      ( 2, datetime( 2026, 9, 2, 12, 30 ), "", "", "Premier", 101 ),
  ]
  assert conn.execute( "SELECT id FROM Person WHERE first_name = 'Same' AND last_name = 'Person'" ).fetchone()[ 0 ] == existing_person_id
  assert conn.execute( "SELECT first_name, last_name FROM Person WHERE first_name = 'Jane'" ).fetchone() == ( "Jane", "Doe Smith" )
  assert conn.execute( "SELECT shirt_number, role FROM PersonParticipation ORDER BY match_id, shirt_number" ).fetchall() == [
      ( 7, "unknown" ),
      ( 12, "unknown" ),
      ( 4, "unknown" ),
      ( 7, "unknown" ),
  ]


@pytest.mark.usefixtures( "clear_test_database" )
def test_import_updates_existing_match_participations_and_adds_late_players( conn ):
  division = DivisionData(
      div=Division( name="Premier", divisionId=10, teamId=20 ),
      matches=[
          FixtureWrapper( match=Fixture(
              id=100,
              date="202609011230",
              players=[ Player( shirt=7, name="Same Person", goals=0, yellows=0, reds=0 ) ],
          ) ),
      ],
  )
  importSquadiDivision( conn, division )
  division.matches[ 0 ].match.players[ 0 ].shirt = 10
  division.matches[ 0 ].match.players.append( Player( shirt=12, name="Late Joiner", goals=0, yellows=0, reds=0 ) )

  result = importSquadiDivision( conn, division, { 100: ( "Updated Home", "Updated Away" )} )

  assert result.matches_created == 0
  assert result.persons_created == 1
  assert result.participations_created == 1
  assert result.participations_updated == 1
  assert conn.execute( "SELECT shirt_number FROM PersonParticipation WHERE person_id = (SELECT id FROM Person WHERE first_name = 'Same')" ).fetchone() == ( 10,)
  assert conn.execute( "SELECT home, away FROM matches WHERE squadi_id = 100" ).fetchone() == ( "Updated Home", "Updated Away" )


@pytest.mark.usefixtures( "clear_test_database" )
def test_import_continues_after_invalid_player_and_reports_error( conn ):
  division = DivisionData(
      div=Division( name="Premier", divisionId=10, teamId=20 ),
      matches=[
          FixtureWrapper(
              match=Fixture(
                  id=100,
                  date="202609011230",
                  players=[
                      Player( shirt=7, name="Valid Player", goals=0, yellows=0, reds=0 ),
                      Player( shirt=8, name="", goals=0, yellows=0, reds=0 ),
                      Player( shirt=9, name="Also Valid", goals=0, yellows=0, reds=0 ),
                  ],
              )
          ),
          FixtureWrapper( match=Fixture(
              id=101,
              date="202609021230",
              players=[ Player( shirt=4, name="Next Match", goals=0, yellows=0, reds=0 ) ],
          ) ),
      ],
  )

  result = importSquadiDivision( conn, division )

  assert result.matches_created == 2
  assert result.persons_created == 3
  assert result.participations_created == 3
  assert len( result.errors ) == 1
  assert "Squadi match 100, player ''" in result.errors[ 0 ]
  assert conn.execute( "SELECT count(*) FROM PersonParticipation" ).fetchone() == ( 3,)


@pytest.mark.usefixtures( "clear_test_database" )
def test_person_and_participation_upserts_and_explicit_roles( conn ):
  match = upsertMatch( conn, Match( id=0, date=datetime( 2026, 9, 1, tzinfo=fixedTZ ), home="A", away="B", division="D" ) )
  person = upsertPerson( conn, Person( id=0, first_name="Goal", last_name="Keeper" ) )
  assert upsertPerson( conn, Person( id=0, first_name="Goal", last_name="Keeper" ) ).id == person.id

  for role in ( "home_player", "home_goalkeeper", "away_player", "away_goalkeeper" ):
    participation = upsertPersonParticipation(
        conn,
        PersonParticipationDB( match_id=match.id, person_id=person.id, shirt_number=1, role=role ),
    )
    assert participation.role == role

  with pytest.raises( duckdb.ConstraintException ):
    conn.execute(
        "UPDATE PersonParticipation SET role = 'home' WHERE match_id = ? AND person_id = ?",
        [ match.id, person.id ],
    )


@pytest.mark.usefixtures( "clear_test_database" )
def test_generic_people_are_created_once_and_assigned_with_generic_roles( conn ):
  match = upsertMatch(
      conn,
      Match( id=0, date=None, home="A", away="B", division="D" ),
  )

  people = ensureGenericPersons( conn )
  repeated_people = ensureGenericPersons( conn )

  assert len( people ) == 25
  assert [ ( person.first_name, person.last_name ) for person in people ] == [
      ( "Main", "Referee" ),
      ( "Assistant", "Referee 1" ),
      ( "Assistant", "Referee 2" ),
      ( "Opposition", "Keeper 1" ),
      ( "Opposition", "Keeper 2" ),
      *( ( "Opposition", f"Player {number}" ) for number in range( 1, 21 ) ),
  ]
  assert [ person.id for person in repeated_people ] == [ person.id for person in people ]
  assert conn.execute( "SELECT count(*) FROM Person" ).fetchone() == ( 25, )

  participations = addGenericPeopleToMatch( conn, match.id )
  repeated_participations = addGenericPeopleToMatch( conn, match.id )

  assert len( participations ) == len( repeated_participations ) == 25
  assert conn.execute(
      "SELECT role, count(*) FROM PersonParticipation GROUP BY role ORDER BY role"
  ).fetchall() == [
      ( "away_goalkeeper", 2 ),
      ( "away_player", 20 ),
      ( "referee", 3 ),
  ]
  assert conn.execute( "SELECT count(*) FROM PersonParticipation WHERE is_placeholder" ).fetchone() == ( 25, )
