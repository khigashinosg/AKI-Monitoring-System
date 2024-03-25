import unittest
from unittest.mock import patch
import sqlite3
import os
import sys
import pickle
import tempfile
import warnings
import statistics
import csv
from sklearn.metrics import fbeta_score

try:
    # Try importing as if running in Docker (without src prefix)
    from prediction_system import *
except ModuleNotFoundError:
    # Fallback to local import with src prefix
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from prediction_system import *


class TestAKIPredictor(unittest.TestCase):
    db_path = None
    db_file = None
    conn = None
    cursor = None
    model = None

    @classmethod
    def setUpClass(cls):
        # Load the actual model from a pickle file
        with open("models/trained_model.pkl", "rb") as file:
            cls.model = pickle.load(file)

        # Create a temporary file to use as the database
        cls.db_file, cls.db_path = tempfile.mkstemp(suffix='.db')
        cls.conn = sqlite3.connect(cls.db_path)
        cls.cursor = cls.conn.cursor()
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="sklearn.*")

        cls.setup_database()

    def setUp(self):
        # Prepare your test environment
        self.aki_predictor = AKIPredictor(self.model, self.db_path,
                                          metrics_count_flag=False)

    @classmethod
    def tearDownClass(cls):
        # Close the connection and remove the temporary database file
        if cls.conn:
            cls.conn.close()
        if cls.db_file:
            os.close(cls.db_file)
        if cls.db_path:
            os.unlink(cls.db_path)

    @classmethod
    def setup_database(cls):
        cls.cursor.execute('''CREATE TABLE patient_history (
                                mrn TEXT PRIMARY KEY,
                                age INTEGER,
                                sex INTEGER,
                                test_1 REAL,
                                test_2 REAL,
                                test_3 REAL,
                                test_4 REAL,
                                test_5 REAL
                              )''')
        test_data = [
            ("640400", 33, 0, 107.66, 116.58, 85.98, 100.95, 104.96),
            ("755374", 50, 1, 112.34, 94.65, 89.37, 98.63, 97.07),
            ("442925", 16, 1, 73.93, 98.37, 82.16, 78.02, 70.88),
            ("160064", 42, 0, 84.54, 88.10, 76.24, 79.46, 83.36),
            ("164125", 24, 0, 104.02, 82.11, 107.74, 107.71, 90.60)
        ]
        cls.cursor.executemany('''INSERT INTO patient_history (mrn, age, sex, 
                                    test_1, test_2, test_3, test_4, test_5)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', test_data)

        cls.conn.commit()

    @patch('prediction_system.logging.error')
    def test_invalid_mrn(self, mock_log_error):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||M89928",  # Invalid MRN
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||127.57"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Invalid MRN should not trigger AKI prediction")

        # Verify that an error was logged
        msg_identifier = \
            "\n[MRN: M89928 \nmessage_type <ORU^R01>\ntimestamp: 20240331003200]"
        mock_log_error.assert_called_once_with(f"{msg_identifier}\n>> "
                                               f"Invalid MRN format: M89928")

        # Verify no entry is created for the new patient due to invalid DOB
        self.cursor.execute("SELECT 1 FROM patient_history WHERE mrn = 'M89928'")
        patient_data = self.cursor.fetchone()
        self.assertIsNone(patient_data, "Patient data should not be created for "
                                        "invalid MRN")

    @patch('prediction_system.logging.error')
    def test_invalid_message_indexing(self, mock_log_error):
        # Message with missing segments to trigger IndexError
        message = ["MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200"]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Invalid message format should not trigger "
                                  "AKI prediction")

        # Verify that an error was logged with correct content
        mock_log_error.assert_called()
        mock_log_error.assert_called_with("Error processing message due to "
                                          "invalid message format: list index "
                                          "out of range")

    @patch('prediction_system.logging.error')
    def test_lims_message_not_for_creatinine(self, mock_log_error):
        # Example message with a test type other than creatinine
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||442925",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|GLUCOSE||100"  # Non-creatinine test result
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Non-creatinine test results should not trigger "
                                  "AKI prediction")

        # Verify that an error was logged
        msg_identifier = \
            "\n[MRN: 442925 \nmessage_type <ORU^R01>\ntimestamp: 20240331003200]"
        mock_log_error.assert_called_once_with(f"{msg_identifier}\n>> "
                                               f"Invalid test type: GLUCOSE")

        # Verify no update for non-creatinine result message
        self.cursor.execute("SELECT test_1, test_2, test_3, test_4, test_5 "
                            "FROM patient_history WHERE mrn = '442925'")
        updated_tests = self.cursor.fetchone()
        expected_tests = (73.93, 98.37, 82.16, 78.02, 70.88)
        self.assertEqual(updated_tests, expected_tests, "Patient data should "
                                                        "not be updated for "
                                                        "non-creatinine results")

    @patch('prediction_system.logging.error')
    def test_incorrect_creatinine_value(self, mock_log_error):
        # An example message with incorrect creatinine test result
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||640400",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||y68.09"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Incorrect creatinine value should not trigger "
                                  "AKI prediction")

        # Verify that an error was logged for incorrect creatinine value
        msg_identifier = \
            "\n[MRN: 640400 \nmessage_type <ORU^R01>\ntimestamp: 20240331003200]"
        mock_log_error.assert_called_with(f"{msg_identifier}\n>> "
                                          f"Invalid test result format: y68.09")

        # Verify no update for invalid creatinine result message
        self.cursor.execute("SELECT test_1, test_2, test_3, test_4, test_5 "
                            "FROM patient_history WHERE mrn = '640400'")
        updated_tests = self.cursor.fetchone()
        expected_tests = (107.66, 116.58, 85.98, 100.95, 104.96)
        self.assertEqual(updated_tests, expected_tests, "Patient data should not "
                                                        "be updated with invalid "
                                                        "creatinine results")

    def test_lims_message_current_patient_updates_entry(self):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||640400",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||110.0"
        ]
        mrn = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(mrn, "MRN should not be returned for negative AKI "
                               "prediction")

        # Verify that the test results have been updated and shifted properly
        self.cursor.execute("SELECT test_1, test_2, test_3, test_4, test_5 "
                            "FROM patient_history WHERE mrn = '640400'")
        updated_tests = self.cursor.fetchone()
        expected_tests = (110.0, 107.66, 116.58, 85.98, 100.95)
        self.assertEqual(updated_tests, expected_tests, "Test results have not "
                                                        "been updated and shifted "
                                                        "properly")

    def test_lims_message_new_patient_creates_entry(self):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||999999",  # New MRN not previously in the database
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||150.0"
        ]
        mrn = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(mrn, "MRN should not be returned for missing "
                               "demographic info where LIMS is received before "
                               "PAS for a new patient")

        # Verify that a new entry was created in the database
        self.cursor.execute(
            "SELECT age, sex, test_1, test_2, test_3, test_4, test_5 "
            "FROM patient_history WHERE mrn = ?", ("999999",))
        patient_data = self.cursor.fetchone()

        expected_data = (None, None, 150.0, 150.0, 150.0, 150.0, 150.0)
        self.assertEqual(patient_data, expected_data, "New patient entry with "
                                                      "test results was not "
                                                      "created correctly")

        # Verify that the MRN is added to pending_predictions
        self.assertIn("999999", self.aki_predictor.pending_predictions,
                      "New patient MRN was not added to pending_predictions")

    @patch('prediction_system.logging.error')
    def test_pas_message_with_invalid_dob(self, mock_log_error):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ADT^A01|||2.5",
            "PID|1||164125||JOHN DOE||20012312|M",  # Patient with invalid DOB
            "NK1|1|ERICA DOE|PARTNER"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Invalid DOB should not trigger any database "
                                  "update or AKI prediction")

        # Verify that an error was logged with the correct message
        msg_identifier = \
            "\n[MRN: 164125 \nmessage_type <ADT^A01>\ntimestamp: 20240331003200]"
        mock_log_error.assert_called_once_with(f"{msg_identifier}\n>> "
                                               f"Invalid DOB format: 20012312")

        # Verify that no update occurs for the invalid dob value in the database
        self.cursor.execute("SELECT age, sex FROM patient_history "
                            "WHERE mrn = '164125'")
        patient_data = self.cursor.fetchone()
        expected_data = (24, 0)
        self.assertEqual(patient_data, expected_data, "Patient data should not be "
                                                      "created for invalid DOB")

    @patch('prediction_system.logging.error')
    def test_pas_message_with_invalid_sex(self, mock_log_error):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ADT^A01|||2.5",
            "PID|1||164125||JANE DOE||20010312|W",  # Patient with invalid sex
            "NK1|1|JOHN DOE|PARTNER"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "Invalid sex should not trigger any database "
                                  "update or AKI prediction")

        # Verify that an error was logged with the correct message
        msg_identifier = \
            "\n[MRN: 164125 \nmessage_type <ADT^A01>\ntimestamp: 20240331003200]"
        mock_log_error.assert_called_once_with(f"{msg_identifier}\n>> "
                                               f"Invalid sex value: W")

        # Verify that no update occurs for the invalid sex value in the database
        self.cursor.execute("SELECT age, sex FROM patient_history "
                            "WHERE mrn = '164125'")
        patient_data = self.cursor.fetchone()
        expected_data = (24, 0)
        self.assertEqual(patient_data, expected_data, "Patient data should not be "
                                                      "created for invalid sex")

    def test_pas_message_updates_age(self):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ADT^A01|||2.5",
            "PID|1||755374||JOHN DOE||19800312|M",
            "NK1|1|ERICA DOE|PARTNER"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "PAS message for MRN not in pending_predictions "
                                  "should return None")

        # Verify that age has been updated in the database
        self.cursor.execute("SELECT age, sex FROM patient_history "
                            "WHERE mrn = '755374'")
        patient_data = self.cursor.fetchone()
        expected_age = self.aki_predictor._calculate_age("19800312")
        expected_data = (expected_age, 0)
        self.assertEqual(patient_data, expected_data, f"Age should be updated to "
                                                      f"{expected_age} in the "
                                                      f"database")

    def test_pas_message_creates_new_patient_record(self):
        message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ADT^A01|||2.5",
            "PID|1||55555||JOHN SMITH||19900101|M",  # New MRN
            "NK1|1|ERICA SMITH|PARTNER"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(message)
        self.assertIsNone(result, "No MRN should be returned for PAS message for "
                                  "new patient")

        # Verify a new record was created with the correct age and sex
        self.cursor.execute(
            "SELECT age, sex, test_1, test_2, test_3, test_4, test_5 "
            "FROM patient_history WHERE mrn = '55555'"
        )
        patient_data = self.cursor.fetchone()

        # Assert the new record exists with the correct age and sex
        self.assertIsNotNone(patient_data, "A new patient record should have "
                                           "been created")
        expected_age = self.aki_predictor._calculate_age('19900101')
        self.assertEqual(patient_data[0], expected_age, "Age should match the "
                                                        "calculated value based "
                                                        "on DOB")
        self.assertEqual(patient_data[1], 0, "Sex should be correctly set as "
                                             "male (0 for M)")
        for test_result in patient_data[2:]:  # Test results 1 to 5
            self.assertIsNone(test_result, "Test results should be initialised "
                                           "to NULL for a new patient")

    def test_lims_then_pas_for_new_patient_updates_and_attempts_aki_prediction(self):
        # Step 1: Simulate LIMS message for a new patient
        lims_message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||333333",  # New MRN not previously in the database
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||100.0"
        ]
        result_lims = self.aki_predictor.examine_message_and_predict_aki(lims_message)
        self.assertIsNone(result_lims, "MRN should not be returned for missing "
                                       "emographic info where LIMS is received "
                                       "before PAS")

        # Verify a new entry was created with correct tests and MRN is in
        # pending_predictions
        self.cursor.execute(
            "SELECT age, sex, test_1, test_2, test_3, test_4, test_5 "
            "FROM patient_history WHERE mrn = '333333'"
        )
        patient_data = self.cursor.fetchone()
        self.assertEqual(patient_data,
                         (None, None, 100.0, 100.0, 100.0, 100.0, 100.0),
                         "LIMS message for a new patient did not correctly "
                         "create entry")
        self.assertIn('333333', self.aki_predictor.pending_predictions,
                      "New patient MRN was not added to pending_predictions")

        # Step 2: Simulate PAS message for the same patient, updating demographic info
        pas_message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ADT^A01|||2.5",
            "PID|1||333333||JOHN DOE||19800312|M",  # Same MRN as LIMS
            "NK1|1|ERICA SMITH|PARTNER"
        ]
        result_pas = self.aki_predictor.examine_message_and_predict_aki(pas_message)
        self.assertIsNone(result_pas, "PAS message should not return MRN when "
                                      "demographic info is updated (for "
                                      "pending_prediction's MRN) without "
                                      "triggering AKI prediction")

        # Verify the database is updated with demographic info and attempt
        # an AKI prediction
        self.cursor.execute(
            "SELECT age, sex, test_1, test_2, test_3, test_4, test_5 "
            "FROM patient_history WHERE mrn = '333333'")
        updated_data = self.cursor.fetchone()
        expected_age = self.aki_predictor._calculate_age("19800312")
        self.assertEqual(updated_data,
                         (expected_age, 0, 100.0, 100.0, 100.0, 100.0, 100.0),
                         "PAS message did not update demographic info correctly")

        # Step 3: Verify that an AKI prediction attempt was made and MRN removed
        # from pending_predictions
        self.assertNotIn('333333', self.aki_predictor.pending_predictions,
                         "MRN should be removed from pending_predictions after "
                         "demographic update and AKI prediction attempt")

    @patch('prediction_system.logging.info')
    def test_detect_aki_high_creatinine_returns_mrn(self, mock_log_info):
        # High creatinine level message for an existing patient
        high_creatinine_message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||160064",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||300"  # High creatinine level, should trigger AKI
            # prediction
        ]
        mrn = self.aki_predictor.examine_message_and_predict_aki(
            high_creatinine_message
        )

        # Check if MRN is correctly returned, indicating a positive AKI prediction
        self.assertIsNotNone(mrn, "MRN should be returned for positive AKI "
                                  "prediction due to high creatinine level")
        self.assertEqual(mrn, "160064", "MRN of the patient with high "
                                        "creatinine level should be returned")

        # Verify that an info log was made with the correct message
        msg_identifier = \
            "\n[MRN: 160064 \nmessage_type <ORU^R01>\ntimestamp: 20240331003200]"
        mock_log_info.assert_called_once_with(f"{msg_identifier}\n>> "
                                              f"AKI predicted for MRN: 160064")

    def test_pas_discharge_message_does_not_update_database(self):
        # Discharge message for non-existing patient
        discharge_message = [
            "MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401201630||ADT^A03|||2.5",
            "PID|1||478237423"
        ]
        result = self.aki_predictor.examine_message_and_predict_aki(
            discharge_message
        )
        self.assertIsNone(result, "PAS discharge message should not trigger "
                                  "AKI prediction")

        # Verify no entry is created in the database for the new patient
        self.cursor.execute("SELECT 1 FROM patient_history "
                            "WHERE mrn = '478237423'")
        patient_data = self.cursor.fetchone()
        self.assertIsNone(patient_data, "Patient data should not be created "
                                        "for discharge message")


class TestMLLPConversion(unittest.TestCase):
    def test_to_mllp(self):
        ACK = [
            "MSH|^~\&|||||20240129093837||ACK|||2.5",
            "MSA|AA",
        ]
        ack = to_mllp(ACK)
        expected_mllp = \
            b'\x0bMSH|^~\\&|||||20240129093837||ACK|||2.5\rMSA|AA\r\x1c\r'
        self.assertEqual(ack, expected_mllp, "The MLLP conversion of ACK message "
                                             "did not match the expected output.")

    def test_from_mllp(self):
        # Test the decoding of MLLP-encoded message back into HL7 segments
        mllp_message = \
            b'\x0bMSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240102135300||' \
            b'ADT^A01|||2.5\rPID|1||497030||ROSCOE DOHERTY||19870515|M\r\x1c\r'
        expected_segments = [
            "MSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240102135300||ADT^A01|||2.5",
            "PID|1||497030||ROSCOE DOHERTY||19870515|M"
        ]
        self.assertEqual(from_mllp(mllp_message), expected_segments,
                         "Decoding from MLLP did not produce the expected HL7 "
                         "message segments.")

        # Test for another MLLP encoded message
        mllp_message_2 = b'\x0bMSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||' \
                         b'20240331003200||ORU^R01|||2.5\rPID|1||125412\rOBR|1|' \
                         b'|||||20240331003200\rOBX|1|SN|CREATININE||' \
                         b'127.5695463720204\r\x1c\r'
        expected_segments_2 = [
            "MSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||125412",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||127.5695463720204"
        ]
        self.assertEqual(from_mllp(mllp_message_2), expected_segments_2,
                         "Decoding from MLLP did not produce the expected HL7 "
                         "message segments for the second test.")

    def test_from_mllp_with_omitted_start_and_end_block(self):
        mllp_message_3 = b'MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||' \
                         b'ORU^R01|||2.5\rPID|1||125412\rOBR|1||||||20240331003200' \
                         b'\rOBX|1|SN|CREATININE||127.5695463720204'
        expected_segments_3 = [
            "SH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240331003200||ORU^R01|||2.5",
            "PID|1||125412",
            "OBR|1||||||20240331003200",
            "OBX|1|SN|CREATININE||127.5695463720"
        ]
        self.assertEqual(from_mllp(mllp_message_3), expected_segments_3,
                         "Decoding from MLLP did not produce the expected HL7 "
                         "message segments for the third test.")

    def test_from_mllp_with_double_carriage_returns(self):
        mllp_message_4 = b'\x0bMSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||' \
                         b'20240331035800||ADT^A03|||2.5\r\rPID|1||829339\r\x1c\r'
        expected_segments_4 = [
            "MSH|^~\\&|SIMULATION|SOUTH RIVERSIDE|||20240331035800||ADT^A03|||2.5",
            "",
            "PID|1||829339"
        ]
        self.assertEqual(from_mllp(mllp_message_4), expected_segments_4,
                         "Decoding from MLLP did not produce the expected HL7 "
                         "message segments for the fourth test.")


class TestPreloadHistoryToSQLite(unittest.TestCase):
    db_path = None
    db_file = None
    conn = None

    @classmethod
    def setUpClass(cls):
        # Create a temporary file to use as the database.
        cls.db_file, cls.db_path = tempfile.mkstemp(suffix='.db')

        # Connect to the temporary file database and preload data.
        cls.conn = sqlite3.connect(cls.db_path)
        history_csv_path = os.getenv("HISTORY_CSV_PATH",
                                     "data/hospital-history/history.csv")
        preload_history_to_sqlite(cls.db_path, history_csv_path)

    @classmethod
    def tearDownClass(cls):
        # Close the database connection and remove the temporary file.
        cls.conn.close()
        os.close(cls.db_file)
        os.remove(cls.db_path)

    def test_table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' "
                       "AND name='patient_history';")
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists, "Table 'patient_history' "
                                           "should exist.")

    def test_initial_data_loaded(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_history;")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0, "Initial data should be loaded into "
                                     "'patient_history' table.")

    def test_column_structure(self):
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(patient_history);")
        columns_info = cursor.fetchall()
        columns = [info[1] for info in columns_info]
        expected_columns = ['mrn', 'age', 'sex', 'test_1', 'test_2', 'test_3',
                            'test_4', 'test_5']
        self.assertEqual(columns, expected_columns, "Table structure does not "
                                                    "match expected columns.")

    def test_age_and_sex_values_none_initially(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT age, sex FROM patient_history;")
        for row in cursor.fetchall():
            age, sex = row
            self.assertIsNone(age, "All 'age' values should initially be None.")
            self.assertIsNone(sex, "All 'sex' values should initially be None.")

    def test_mrn_and_test_result_values_not_none_initially(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT mrn, test_1, test_2, test_3, test_4, test_5 "
                       "FROM patient_history;")
        rows = cursor.fetchall()
        self.assertTrue(rows, "The database should have patient records after "
                              "preloading.")
        for row in rows:
            mrn, test_1, test_2, test_3, test_4, test_5 = row
            self.assertTrue(mrn.isdigit(), f"MRN {mrn} is not all digit.")

            # Check each test result to ensure it can be represented as a float.
            for idx, test_result in enumerate(
                    [test_1, test_2, test_3, test_4, test_5], start=1):
                try:
                    float(test_result)
                except ValueError:
                    self.fail(f"Test result {test_result} for MRN {mrn}, "
                              f"test_{idx} cannot be converted to float.")

    def test_database_persistence_across_reconnections(self):
        db_path = self.db_path

        # Insert a new record
        def insert_record(conn):
            insert_query = """INSERT INTO patient_history (mrn, age, sex, 
                              test_1, test_2, test_3, test_4, test_5) 
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            test_record = ("1223334444",
                           40, 1, 100.0, 100.0, 100.0, 100.0, 100.0)
            conn.execute(insert_query, test_record)
            conn.commit()

        # Check the record exists
        def check_record_exists(conn):
            select_query = "SELECT * FROM patient_history WHERE mrn = ?"
            cursor = conn.cursor()
            cursor.execute(select_query, ("1223334444",))
            return cursor.fetchone()

        # Insert record in a fresh connection
        conn1 = sqlite3.connect(db_path)
        insert_record(conn1)
        conn1.close()  # Close the connection after inserting

        # Reopen the connection to check for persistence
        conn2 = sqlite3.connect(db_path)
        fetched_record = check_record_exists(conn2)
        self.assertIsNotNone(fetched_record, "The record should persist after "
                                             "reconnecting to the database.")
        conn2.close()  # Close the connection after checking

        # Clean up by removing the test record in a new connection
        conn3 = sqlite3.connect(db_path)
        conn3.execute("DELETE FROM patient_history WHERE mrn = ?",
                      ("1223334444",))
        conn3.commit()
        conn3.close()


class TestModelF3Score(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the model
        with open("models/trained_model.pkl", "rb") as file:
            cls.model = pickle.load(file)

        # Environment variables for test data and labels paths
        test_data_path = os.getenv("TEST_DATA_PATH",
                                   "data/test_data/test_f3.csv")
        labels_path = os.getenv("LABELS_PATH",
                                "data/test_data/labels_f3.csv")

        # Load and preprocess test data and labels
        cls.X_test, cls.y_test = \
            cls.load_and_preprocess_test_data(test_data_path, labels_path)

    @staticmethod
    def load_and_preprocess_test_data(test_data_path, labels_path):
        # Process the test data
        with open(test_data_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row

            X_test = []
            for row in csv_reader:
                cleaned_row = [value for value in row if value != '']
                age = int(cleaned_row[0])
                sex = 0 if cleaned_row[1].lower() == 'm' else 1
                test_results = list(map(float, cleaned_row[3::2]))
                test_results = test_results[::-1]

                # If there are fewer than 5 test results, pad with the mean
                while len(test_results) < 5:
                    average_result = statistics.mean(test_results) \
                        if test_results else 0
                    test_results += [average_result] * (5 - len(test_results))

                test_results = test_results[:5]

                X_test.append([age, sex] + test_results)

        # Process the labels
        with open(labels_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            y_test = [row for row in csv_reader]

        # Convert labels to binary (0 or 1)
        y_test = [0 if label[0].lower() == 'n' else 1 for label in y_test]

        return X_test, y_test

    def test_f3_score(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate the F3 score
        f3_score = fbeta_score(self.y_test, y_pred, beta=3)

        # Verify F3 score meets a min. threshold of 70% (better than NHS algo.)
        self.assertGreaterEqual(f3_score, 0.7, f"F3 score is below the 70% "
                                               f"threshold: {f3_score}")


if __name__ == '__main__':
    unittest.main()
