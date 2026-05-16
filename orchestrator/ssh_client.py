import paramiko
import time
import socket
from pathlib import Path


SSH_RECOVERABLE_ERRORS = (
    paramiko.SSHException,
    paramiko.ssh_exception.NoValidConnectionsError,
    EOFError,
    OSError,
    socket.timeout,
)


class DUTClient:
    def __init__(
        self,
        hostname,
        username,
        root_access=False,
        key_filename=None,
        backup_ip=None,
        keepalive_sec=30,
    ):
        self.hostname = hostname
        self.username = username
        self.backup_ip = backup_ip
        self.root_access = root_access
        self.key_filename = key_filename
        self.keepalive_sec = keepalive_sec

        self.user = None
        self.root = None
        self.connected_host = None

    def _connect_single(self, hostname, username, timeout):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        client.connect(
            hostname=hostname,
            username=username,
            key_filename=self.key_filename,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )

        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(self.keepalive_sec)

        return client

    def _client_is_active(self, client):
        if client is None:
            return False

        transport = client.get_transport()
        if transport is None:
            return False

        return transport.is_active() and transport.is_authenticated()

    def _close_client(self, client):
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def close(self):
        self._close_client(self.user)
        self._close_client(self.root)

        self.user = None
        self.root = None
        self.connected_host = None

    def connect(self, timeout=20, force=False):
        if force:
            self.close()
        else:
            user_ok = self._client_is_active(self.user)
            root_ok = True

            if self.root_access:
                root_ok = self._client_is_active(self.root)

            if user_ok and root_ok:
                return

            self.close()

        targets = [self.hostname]
        if self.backup_ip and self.backup_ip != self.hostname:
            targets.append(self.backup_ip)

        last_exc = None

        for target in targets:
            try:
                user_client = self._connect_single(
                    hostname=target,
                    username=self.username,
                    timeout=timeout,
                )

                root_client = None
                if self.root_access:
                    root_client = self._connect_single(
                        hostname=target,
                        username="root",
                        timeout=timeout,
                    )

                self.user = user_client
                self.root = root_client
                self.connected_host = target
                return

            except Exception as e:
                last_exc = e

                self._close_client(self.user)
                self._close_client(self.root)

                self.user = None
                self.root = None
                self.connected_host = None

        raise ConnectionError(
            f"Failed to connect to primary host '{self.hostname}'"
            + (f" or backup IP '{self.backup_ip}'" if self.backup_ip else "")
        ) from last_exc

    def _get_client(self, as_root=False):
        if as_root:
            if not self.root_access:
                raise RuntimeError("Root SSH access was requested, but root_access=False")

            return self.root

        return self.user

    def _ensure_connected(self, as_root=False, timeout=20):
        client = self._get_client(as_root)

        if not self._client_is_active(client):
            self.connect(timeout=timeout, force=True)

        client = self._get_client(as_root)

        if not self._client_is_active(client):
            raise RuntimeError("SSH client not connected after reconnect attempt")

        return client

    def run(
        self,
        cmd,
        as_root=False,
        async_run=False,
        retries=2,
        timeout=20,
        safe_to_retry=True,
    ):
        """
        Run a command on the DUT.

        safe_to_retry=False should be used for commands that must not run twice,
        such as starting a benchmark in the background.
        """

        if not safe_to_retry:
            retries = 1

        last_exc = None

        for attempt in range(1, retries + 1):
            try:
                client = self._ensure_connected(as_root=as_root, timeout=timeout)

                stdin, stdout, stderr = client.exec_command(cmd)

                if async_run:
                    return None, None, None

                # For small command outputs this is fine.
                # If you ever run commands with huge output, this should become
                # a streaming read loop to avoid SSH window backpressure.
                out = stdout.read().decode(errors="replace")
                err = stderr.read().decode(errors="replace")
                code = stdout.channel.recv_exit_status()

                return code, out, err

            except SSH_RECOVERABLE_ERRORS as e:
                last_exc = e

                self.close()

                if attempt >= retries:
                    break

                time.sleep(1)

        raise RuntimeError(f"SSH command failed after {retries} attempt(s): {cmd}") from last_exc

    def reboot(self):
        try:
            self.run("reboot", as_root=True, safe_to_retry=False)
        except Exception:
            # SSH disconnect during reboot is expected.
            pass

    def wait_for_ssh(self, timeout_sec=300, retry_interval=5):
        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            try:
                self.connect(timeout=10, force=True)
                return True
            except Exception:
                time.sleep(retry_interval)

        return False

    def fetch_directory(self, remote_dir, local_dir, as_root=None, retries=2):
        """
        Fetch all files directly inside remote_dir.

        This mirrors your current behavior: it does not recurse into subdirectories.
        """

        if as_root is None:
            as_root = self.root_access

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        last_exc = None

        for attempt in range(1, retries + 1):
            try:
                client = self._ensure_connected(as_root=as_root)
                sftp = client.open_sftp()

                try:
                    for entry in sftp.listdir(remote_dir):
                        remote_path = f"{remote_dir}/{entry}"
                        local_path = local_dir / entry
                        sftp.get(remote_path, str(local_path))
                finally:
                    sftp.close()

                return

            except SSH_RECOVERABLE_ERRORS as e:
                last_exc = e
                self.close()

                if attempt < retries:
                    time.sleep(1)

        raise RuntimeError(f"Failed to fetch remote directory: {remote_dir}") from last_exc