"""
Industrial-Grade Security and Compliance Framework
=================================================

Comprehensive security measures and compliance features for production ML pipelines.
"""

import os
import hashlib
import hmac
import secrets
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DataClassification(Enum):
    """Data classification types"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"
    FINANCIAL = "financial"
    HEALTH = "health"

class UserRole(Enum):
    """User roles for access control"""
    VIEWER = "viewer"
    ANALYST = "analyst"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    require_encryption: bool = True
    require_authentication: bool = True
    require_authorization: bool = True
    require_audit_logging: bool = True
    data_retention_days: int = 365
    max_login_attempts: int = 3
    session_timeout_minutes: int = 60
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    
@dataclass 
class AuditLogEntry:
    """Audit log entry"""
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

class DataEncryption:
    """
    Data encryption utilities for sensitive information
    """
    
    def __init__(self, password: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password"""
        password_bytes = password.encode()
        salt = b'salt_'  # In production, use a proper random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file"""
        file_path = Path(file_path)
        output_path = Path(output_path) if output_path else file_path.with_suffix(file_path.suffix + '.enc')
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.cipher.encrypt(file_data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"File encrypted: {file_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file"""
        encrypted_file_path = Path(encrypted_file_path)
        
        if not output_path:
            # Remove .enc extension if present
            if encrypted_file_path.suffix == '.enc':
                output_path = encrypted_file_path.with_suffix('')
            else:
                output_path = encrypted_file_path.with_suffix('.dec')
        
        output_path = Path(output_path)
        
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            self.logger.info(f"File decrypted: {encrypted_file_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise

class AccessControlManager:
    """
    Role-based access control for ML pipeline resources
    """
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # User sessions and permissions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_roles: Dict[str, UserRole] = {}
        self.failed_login_attempts: Dict[str, int] = {}
        
        # Resource permissions
        self.role_permissions = self._initialize_role_permissions()
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[str]]:
        """Initialize role-based permissions"""
        return {
            UserRole.VIEWER: {
                'read_data', 'view_models', 'view_reports'
            },
            UserRole.ANALYST: {
                'read_data', 'view_models', 'view_reports', 
                'run_analysis', 'create_reports'
            },
            UserRole.DATA_SCIENTIST: {
                'read_data', 'write_data', 'view_models', 'create_models',
                'view_reports', 'run_analysis', 'create_reports'
            },
            UserRole.ML_ENGINEER: {
                'read_data', 'write_data', 'view_models', 'create_models',
                'deploy_models', 'manage_models', 'view_reports', 
                'run_analysis', 'create_reports', 'manage_pipeline'
            },
            UserRole.ADMIN: {
                'read_data', 'write_data', 'view_models', 'create_models',
                'deploy_models', 'manage_models', 'view_reports',
                'run_analysis', 'create_reports', 'manage_pipeline',
                'manage_users', 'view_audit_logs'
            },
            UserRole.SUPER_ADMIN: {
                'read_data', 'write_data', 'view_models', 'create_models',
                'deploy_models', 'manage_models', 'view_reports',
                'run_analysis', 'create_reports', 'manage_pipeline',
                'manage_users', 'view_audit_logs', 'manage_security',
                'system_admin'
            }
        }
    
    def authenticate_user(self, user_id: str, password: str, ip_address: str) -> Optional[str]:
        """
        Authenticate user and create session
        
        Args:
            user_id: User identifier
            password: User password
            ip_address: Client IP address
            
        Returns:
            Session token if successful, None otherwise
        """
        # Check failed login attempts
        if self.failed_login_attempts.get(user_id, 0) >= self.policy.max_login_attempts:
            self.logger.warning(f"User {user_id} exceeded max login attempts")
            return None
        
        # Check IP restrictions
        if self.policy.allowed_ip_ranges and not self._is_ip_allowed(ip_address):
            self.logger.warning(f"Access denied from IP {ip_address}")
            return None
        
        # Simulate password check (in production, use proper authentication)
        if self._verify_password(user_id, password):
            # Reset failed attempts
            self.failed_login_attempts.pop(user_id, None)
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            session_data = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'ip_address': ip_address
            }
            
            self.active_sessions[session_token] = session_data
            
            self.logger.info(f"User {user_id} authenticated successfully")
            return session_token
        else:
            # Increment failed attempts
            self.failed_login_attempts[user_id] = self.failed_login_attempts.get(user_id, 0) + 1
            self.logger.warning(f"Authentication failed for user {user_id}")
            return None
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify user password (placeholder implementation)"""
        # In production, use proper password hashing and verification
        return True  # Placeholder
    
    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is in allowed ranges"""
        # Simplified IP check - in production, use proper CIDR matching
        if not self.policy.allowed_ip_ranges:
            return True
        
        return any(ip_address.startswith(range_prefix) 
                  for range_prefix in self.policy.allowed_ip_ranges)
    
    def authorize_action(self, session_token: str, action: str) -> bool:
        """
        Check if user is authorized for specific action
        
        Args:
            session_token: User session token
            action: Action to authorize
            
        Returns:
            Authorization status
        """
        # Validate session
        if not self._validate_session(session_token):
            return False
        
        session_data = self.active_sessions[session_token]
        user_id = session_data['user_id']
        
        # Get user role
        user_role = self.user_roles.get(user_id, UserRole.VIEWER)
        
        # Check permissions
        allowed_actions = self.role_permissions.get(user_role, set())
        
        is_authorized = action in allowed_actions
        
        if not is_authorized:
            self.logger.warning(f"User {user_id} not authorized for action: {action}")
        
        return is_authorized
    
    def _validate_session(self, session_token: str) -> bool:
        """Validate session token and check expiry"""
        if session_token not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[session_token]
        
        # Check session timeout
        session_age = datetime.now() - session_data['last_activity']
        if session_age > timedelta(minutes=self.policy.session_timeout_minutes):
            self.logout_user(session_token)
            return False
        
        # Update last activity
        session_data['last_activity'] = datetime.now()
        return True
    
    def logout_user(self, session_token: str):
        """Logout user and invalidate session"""
        if session_token in self.active_sessions:
            user_id = self.active_sessions[session_token]['user_id']
            del self.active_sessions[session_token]
            self.logger.info(f"User {user_id} logged out")
    
    def set_user_role(self, user_id: str, role: UserRole):
        """Set user role"""
        self.user_roles[user_id] = role
        self.logger.info(f"User {user_id} assigned role: {role.value}")

class DataPrivacyManager:
    """
    Data privacy and compliance management
    """
    
    def __init__(self, encryption: DataEncryption):
        self.encryption = encryption
        self.logger = logging.getLogger(__name__)
        
        # Data classification cache
        self.data_classifications: Dict[str, DataClassification] = {}
        
        # PII detection patterns (simplified)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
    
    def classify_data(self, data: pd.DataFrame, dataset_name: str) -> DataClassification:
        """
        Automatically classify data sensitivity level
        
        Args:
            data: Dataset to classify
            dataset_name: Name of the dataset
            
        Returns:
            Data classification level
        """
        classification = DataClassification.INTERNAL  # Default
        
        # Check for PII indicators
        column_names = [col.lower() for col in data.columns]
        
        # High sensitivity indicators
        high_sensitivity_indicators = [
            'ssn', 'social_security', 'credit_card', 'medical', 'health',
            'diagnosis', 'treatment', 'salary', 'income', 'financial'
        ]
        
        if any(indicator in ' '.join(column_names) for indicator in high_sensitivity_indicators):
            classification = DataClassification.SENSITIVE_PERSONAL
        
        # Medium sensitivity indicators
        elif any(indicator in ' '.join(column_names) 
                for indicator in ['email', 'phone', 'address', 'name']):
            classification = DataClassification.PERSONAL
        
        # Sample data for content analysis
        if len(data) > 0:
            sample_text = ' '.join(str(val) for val in data.iloc[0].values if pd.notna(val))
            
            for pii_type, pattern in self.pii_patterns.items():
                import re
                if re.search(pattern, sample_text):
                    if classification.value < DataClassification.PERSONAL.value:
                        classification = DataClassification.PERSONAL
                    break
        
        self.data_classifications[dataset_name] = classification
        self.logger.info(f"Dataset {dataset_name} classified as {classification.value}")
        
        return classification
    
    def anonymize_data(self, data: pd.DataFrame, classification: DataClassification) -> pd.DataFrame:
        """
        Anonymize data based on classification level
        
        Args:
            data: Data to anonymize
            classification: Data classification level
            
        Returns:
            Anonymized dataset
        """
        if classification in [DataClassification.PUBLIC, DataClassification.INTERNAL]:
            return data.copy()
        
        anonymized_data = data.copy()
        
        # Apply anonymization based on classification
        if classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]:
            anonymized_data = self._anonymize_pii(anonymized_data)
        
        if classification == DataClassification.SENSITIVE_PERSONAL:
            # Additional anonymization for highly sensitive data
            anonymized_data = self._apply_differential_privacy(anonymized_data)
        
        self.logger.info(f"Data anonymized for classification level: {classification.value}")
        return anonymized_data
    
    def _anonymize_pii(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove or hash personally identifiable information"""
        anonymized = data.copy()
        
        # Identify potential PII columns
        pii_columns = []
        for col in data.columns:
            col_lower = col.lower() 
            if any(pii_indicator in col_lower 
                  for pii_indicator in ['email', 'phone', 'ssn', 'name', 'address']):
                pii_columns.append(col)
        
        # Hash PII columns
        for col in pii_columns:
            if col in anonymized.columns:
                anonymized[col] = anonymized[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:10] if pd.notna(x) else x
                )
        
        return anonymized
    
    def _apply_differential_privacy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply differential privacy (simplified implementation)"""
        anonymized = data.copy()
        
        # Add noise to numeric columns
        numeric_columns = anonymized.select_dtypes(include=['number']).columns
        
        for col in numeric_columns:
            if anonymized[col].notna().sum() > 0:
                noise_scale = anonymized[col].std() * 0.1  # 10% of standard deviation
                noise = np.random.normal(0, noise_scale, len(anonymized))
                anonymized.loc[anonymized[col].notna(), col] += noise
        
        return anonymized
    
    def encrypt_sensitive_data(self, data: pd.DataFrame, 
                             classification: DataClassification) -> pd.DataFrame:
        """
        Encrypt sensitive columns based on classification
        
        Args:
            data: Data to encrypt
            classification: Data classification level
            
        Returns:
            Data with encrypted sensitive columns
        """
        if classification in [DataClassification.PUBLIC, DataClassification.INTERNAL]:
            return data.copy()
        
        encrypted_data = data.copy()
        
        # Identify columns to encrypt
        sensitive_columns = []
        for col in data.columns:
            col_lower = col.lower()
            if any(indicator in col_lower 
                  for indicator in ['ssn', 'credit_card', 'medical', 'financial']):
                sensitive_columns.append(col)
        
        # Encrypt sensitive columns
        for col in sensitive_columns:
            if col in encrypted_data.columns:
                encrypted_data[col] = encrypted_data[col].apply(
                    lambda x: self.encryption.encrypt_data(str(x)) if pd.notna(x) else x
                )
        
        self.logger.info(f"Encrypted {len(sensitive_columns)} sensitive columns")
        return encrypted_data

class AuditLogger:
    """
    Comprehensive audit logging for compliance
    """
    
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def log_action(self, 
                  user_id: str,
                  action: str, 
                  resource: str,
                  result: str,
                  ip_address: Optional[str] = None,
                  additional_data: Optional[Dict[str, Any]] = None):
        """
        Log user action for audit trail
        
        Args:
            user_id: User performing the action
            action: Action performed
            resource: Resource accessed
            result: Result of the action (success, failure, denied)
            ip_address: Client IP address
            additional_data: Additional context data
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            additional_data=additional_data or {}
        )
        
        # Write to audit log file
        log_line = json.dumps({
            'timestamp': entry.timestamp.isoformat(),
            'user_id': entry.user_id,
            'action': entry.action,
            'resource': entry.resource,
            'result': entry.result,
            'ip_address': entry.ip_address,
            'additional_data': entry.additional_data
        })
        
        with open(self.log_file_path, 'a') as f:
            f.write(log_line + '\n')
        
        self.logger.info(f"Audit log: {user_id} {action} {resource} -> {result}")
    
    def get_audit_logs(self, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      user_id: Optional[str] = None) -> List[AuditLogEntry]:
        """
        Retrieve audit logs with optional filtering
        
        Args:
            start_date: Filter from this date
            end_date: Filter to this date  
            user_id: Filter by user ID
            
        Returns:
            List of audit log entries
        """
        entries = []
        
        if not self.log_file_path.exists():
            return entries
        
        with open(self.log_file_path, 'r') as f:
            for line in f:
                try:
                    log_data = json.loads(line.strip())
                    entry_time = datetime.fromisoformat(log_data['timestamp'])
                    
                    # Apply filters
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                    if user_id and log_data['user_id'] != user_id:
                        continue
                    
                    entry = AuditLogEntry(
                        timestamp=entry_time,
                        user_id=log_data['user_id'],
                        action=log_data['action'],
                        resource=log_data['resource'],
                        result=log_data['result'],
                        ip_address=log_data.get('ip_address'),
                        additional_data=log_data.get('additional_data', {})
                    )
                    
                    entries.append(entry)
                    
                except json.JSONDecodeError:
                    continue
        
        return entries

class SecurityManager:
    """
    Main security manager orchestrating all security components
    """
    
    def __init__(self, 
                 policy: SecurityPolicy,
                 audit_log_path: str,
                 encryption_password: Optional[str] = None):
        
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.encryption = DataEncryption(encryption_password)
        self.access_control = AccessControlManager(policy)
        self.privacy_manager = DataPrivacyManager(self.encryption)
        self.audit_logger = AuditLogger(audit_log_path)
    
    def secure_data_processing(self, 
                             data: pd.DataFrame,
                             dataset_name: str,
                             user_session: str,
                             operation: str) -> Optional[pd.DataFrame]:
        """
        Secure data processing pipeline
        
        Args:
            data: Input data
            dataset_name: Name of the dataset
            user_session: User session token
            operation: Operation being performed
            
        Returns:
            Processed data or None if access denied
        """
        user_id = self.access_control.active_sessions.get(user_session, {}).get('user_id', 'unknown')
        
        try:
            # Check authorization
            if not self.access_control.authorize_action(user_session, 'read_data'):
                self.audit_logger.log_action(user_id, operation, dataset_name, 'denied')
                return None
            
            # Classify data
            classification = self.privacy_manager.classify_data(data, dataset_name)
            
            # Apply security measures based on classification
            processed_data = data.copy()
            
            if classification in [DataClassification.PERSONAL, DataClassification.SENSITIVE_PERSONAL]:
                # Anonymize if required
                if self.policy.require_encryption:
                    processed_data = self.privacy_manager.anonymize_data(processed_data, classification)
                
                # Encrypt sensitive data
                processed_data = self.privacy_manager.encrypt_sensitive_data(processed_data, classification)
            
            # Log successful access
            self.audit_logger.log_action(
                user_id, operation, dataset_name, 'success',
                additional_data={
                    'classification': classification.value,
                    'rows': len(data),
                    'columns': len(data.columns)
                }
            )
            
            return processed_data
            
        except Exception as e:
            self.audit_logger.log_action(user_id, operation, dataset_name, 'failure',
                                       additional_data={'error': str(e)})
            self.logger.error(f"Secure data processing failed: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize security policy
    policy = SecurityPolicy(
        require_encryption=True,
        require_authentication=True,
        require_authorization=True,
        require_audit_logging=True,
        max_login_attempts=3,
        session_timeout_minutes=60
    )
    
    # Initialize security manager
    security_manager = SecurityManager(
        policy=policy,
        audit_log_path="logs/audit.log",
        encryption_password="secure_password_123"
    )
    
    # Example user authentication
    session_token = security_manager.access_control.authenticate_user(
        user_id="data_scientist_1",
        password="password",
        ip_address="192.168.1.100"
    )
    
    if session_token:
        # Set user role
        security_manager.access_control.set_user_role("data_scientist_1", UserRole.DATA_SCIENTIST)
        
        # Example secure data processing
        import pandas as pd
        sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
            'age': [25, 30, 35],
            'income': [50000, 75000, 100000]
        })
        
        processed_data = security_manager.secure_data_processing(
            data=sample_data,
            dataset_name="customer_data",
            user_session=session_token,
            operation="data_analysis"
        )
        
        if processed_data is not None:
            print("Data processed securely")
            print(f"Processed data shape: {processed_data.shape}")
        else:
            print("Access denied or processing failed")
    
    else:
        print("Authentication failed")
