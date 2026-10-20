use pyo3::PyErr;
use pyo3::exceptions::PyIOError;

/// Error of reading or writing a file: an array, sky model, calibrator or beam file.
///
/// In Python it is raised as an `IOError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Error {
            message: message.into(),
        }
    }

    /// Returns the description of the error.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<Error> for PyErr {
    fn from(error: Error) -> Self {
        PyIOError::new_err(error.message)
    }
}
