#!/usr/bin/env python3
"""Synthetic local integration checks. No customer files or paid model calls."""
import json
import argparse
import os
import time
import uuid
from io import BytesIO
import httpx
import psycopg
from PIL import Image, ImageDraw, ImageFont
from local_config import load_env


def synthetic_pdf():
    content=b"BT /F1 16 Tf 50 730 Td (SYNTHETIC TEST ONLY 10-1234567) Tj 0 -30 Td (Product evidence sample, not a real clinical report.) Tj ET"
    objects=[b"<< /Type /Catalog /Pages 2 0 R >>",b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",b"<< /Length "+str(len(content)).encode()+b" >>\nstream\n"+content+b"\nendstream"]
    raw=b"%PDF-1.4\n";offsets=[]
    for n, obj in enumerate(objects,1):
        offsets.append(len(raw));raw+=str(n).encode()+b" 0 obj\n"+obj+b"\nendobj\n"
    xref=len(raw);raw+=b"xref\n0 6\n0000000000 65535 f \n"
    raw+=b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return raw+f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


def synthetic_image():
    image=Image.new("RGB",(848,3000),"white")
    draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=32)
    for y in (100,1500,2800):
        draw.text((40,y),"SYNTHETIC ADVERTISEMENT ONLY",font=font,fill="black")
    output=BytesIO();image.save(output,format="PNG");return output.getvalue()


def main(require_queue=False):
    base=os.getenv("MVP_API_URL","http://127.0.0.1:8080")+"/api/v1"
    client=httpx.Client(base_url=base,timeout=45)
    checks=[];suffix=uuid.uuid4().hex[:10]
    password="Synthetic-Only-Password-2026"
    def call(method,path,token=None,**kwargs):
        headers={"Authorization":f"Bearer {token}"} if token else {}
        response=client.request(method,path,headers=headers,**kwargs)
        return response
    def signup(label):
        response=call("POST","/users/signup",json={"email":f"{label}-{suffix}@example.test","password":password,"name":label,"tenantName":f"Synthetic {label} {suffix}"})
        assert response.status_code==200,("signup",response.status_code,response.text)
        return response.json()
    assert call("GET","/reviews").status_code==401
    admin=signup("admin");other=signup("other");token=admin["accessToken"]
    for role in ("author","reviewer"):
        response=call("POST","/users/team",token,json={"email":f"{role}-{suffix}@example.test","password":password,"name":role,"role":role})
        assert response.status_code==200,response.text
    def login(role):
        response=call("POST","/users/login",json={"email":f"{role}-{suffix}@example.test","password":password})
        assert response.status_code==200,response.text
        return response.json()["accessToken"]
    author=login("author");reviewer=login("reviewer")
    checks.append("server JWT and controlled tenant roles")
    author_identity=call("GET","/users/me",author).json()["user"]
    credit_config=call("GET","/billing/config",author)
    assert credit_config.status_code==200,credit_config.text
    review_cost=credit_config.json()["reviewCost"]
    assert isinstance(review_cost,int) and review_cost>0,review_cost
    if "APP_DB_PASSWORD" not in os.environ:
        load_env()
    with psycopg.connect(host="127.0.0.1",port=int(os.getenv("POSTGRES_PORT","55432")),
            dbname=os.getenv("POSTGRES_DB","beauty_capstone"),user="app_user",
            password=os.environ["APP_DB_PASSWORD"],connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE app.users SET credit_balance=%s "
                "WHERE id=%s::uuid AND tenant_id=%s::uuid AND email=%s AND credit_balance=0 RETURNING id",
                (2*review_cost,author_identity["id"],author_identity["tenantId"],
                    f"author-{suffix}@example.test"))
            assert cursor.fetchone() is not None,"Synthetic author credit setup failed"
    def balance():
        response=call("GET","/billing/balance",author)
        assert response.status_code==200,response.text
        return response.json()["balance"]
    assert balance()==2*review_cost
    checks.append("synthetic author funded for two reviews at server-configured cost")
    metadata={"product":{"name":"Synthetic-only product","functionalReportNo":"SYNTHETIC"},"media":[],"evidence":[{"docType":"other"}]}
    image=synthetic_image();pdf=synthetic_pdf()
    def upload(path):
        response=call("POST",path,author,data={"metadata":json.dumps(metadata)},files=[
            ("files",("synthetic.png",image,"image/png")),("files",("synthetic.pdf",pdf,"application/pdf")),
            ("evidence",("synthetic-evidence.pdf",pdf,"application/pdf"))])
        assert response.status_code==202,(response.status_code,response.text)
        return response.json()
    review=upload("/reviews");rid=review["id"]
    def wait(no):
        deadline=time.monotonic()+100
        while time.monotonic()<deadline:
            response=call("GET",f"/reviews/{rid}",author)
            assert response.status_code==200,response.text
            doc=response.json();version=doc["versions"][-1]
            if version["no"]==no and version["status"] in {"done","failed"}:
                assert version["status"]=="done",version
                return doc
            time.sleep(1)
        raise AssertionError("Review did not finish within synthetic smoke timeout")
    def verify_queue(no,kind="run"):
        if not require_queue:
            return
        deadline=time.monotonic()+15
        row=None
        while time.monotonic()<deadline:
            with psycopg.connect(host="127.0.0.1",port=int(os.getenv("POSTGRES_PORT","55432")),
                    dbname=os.getenv("POSTGRES_DB","beauty_capstone"),user="app_user",
                    password=os.environ["APP_DB_PASSWORD"],connect_timeout=5) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT status,attempts,last_error FROM app.review_jobs "
                        "WHERE review_id=%s::uuid AND version_no=%s AND kind=%s",(rid,no,kind))
                    row=cursor.fetchone()
            if row and row[0]=="done":
                assert row[1]==1 and row[2] is None,(no,row)
                return
            time.sleep(1)
        raise AssertionError(f"Queue {kind} job for version {no} did not finish: {row}")
    def wait_resume(no):
        deadline=time.monotonic()+100
        approval=None
        while time.monotonic()<deadline:
            response=call("GET",f"/reviews/{rid}",author)
            assert response.status_code==200,response.text
            version=next(v for v in response.json()["versions"] if v["no"]==no)
            approval=version["approval"]
            assert approval["state"]=="approved",approval
            state=approval.get("agentResume")
            if state=="completed":
                return
            assert state=="pending",("Unexpected resume state",approval)
            time.sleep(1)
        raise AssertionError(f"Agent resume for version {no} timed out: {approval}")
    try:
        review=wait(1);version=review["versions"][-1];result=version["result"]
        verify_queue(1)
        assert balance()==2*review_cost,"Unconfigured analysis did not return the reserved credits"
        assert result["texts"] and any("SYNTHETIC" in text["text"] for text in result["texts"])
        # Without a model key no claim is extracted, so the legal search step has nothing to search.
        expected={"skipped"} if not result["findings"] else {"connected","not_indexed"}
        assert result["retrievalStatus"] in expected and result["legalStatus"]=="not_evaluated",result["retrievalStatus"]
        assert result["visualFindings"]==[] and result["pipelineStatus"]["visual"]=="not_configured"
        assert result["evidence"]["text"] and result["publicData"]["reportNo"]=="SYNTHETIC" and result["publicData"]["status"],result["publicData"]
        assert result["patents"] and result["patents"][0]["id"]=="10-1234567",result["patents"]
        assert "data:image" not in json.dumps(review)
        assert "runToken" not in json.dumps(review)
        checks.append("real PNG and PDF bytes, local OCR, evidence via authenticated MCP")
        assert call("GET",f"/reviews/{rid}",other["accessToken"]).status_code==404
        assert call("POST",f"/reviews/{rid}/decision",author,json={"versionNo":1,"state":"approved","comment":"synthetic"}).status_code==403
        assert call("POST",f"/reviews/{rid}/decision",reviewer,json={"versionNo":1,"state":"approved","comment":""}).status_code==400
        response=call("POST",f"/reviews/{rid}/decision",reviewer,json={"versionNo":1,"state":"approved","comment":"Synthetic manual check; no legal basis was attached."})
        assert response.status_code==200,response.text
        approval=response.json()["versions"][-1]["approval"]
        assert approval["state"]=="approved",approval
        assert approval.get("agentResume") in {"pending","completed"},approval
        wait_resume(1)
        verify_queue(1,"resume")
        assert balance()==2*review_cost,"Agent resume changed the settled credits"
        checks.append("tenant isolation, role check, persisted HITL and PostgreSQL resume")
        upload(f"/reviews/{rid}/re-review");review=wait(2)
        verify_queue(2)
        assert balance()==2*review_cost,"Unconfigured re-review did not return the reserved credits"
        assert review["versions"][0]["approval"]["state"]=="approved"
        assert review["versions"][1]["approval"]["state"]=="pending"
        assert call("POST",f"/reviews/{rid}/decision",reviewer,json={"versionNo":1,"state":"approved","comment":"old version"}).status_code==409
        checks.append("new version with independent approval history")
        sources = call("GET", "/reference-sources", token).json()["sources"]
        assert {"S05", "S06", "S10"} <= {source["id"] for source in sources}
        assert call("DELETE","/users/logout",other["accessToken"]).status_code==200
        assert call("GET","/users/me",other["accessToken"]).status_code==401
        checks.append("handoff source manifest and revoked logout token")
    finally:
        response=call("DELETE",f"/reviews/{rid}",author)
        assert response.status_code==200,("cleanup",response.status_code,response.text)
    assert call("GET",f"/reviews/{rid}",author).status_code==404
    checks.append("review deletion")
    if require_queue:
        checks.append("RabbitMQ-backed run jobs for both versions and the approval resume job reached done once")
    print(json.dumps({"checks":checks,"paid_model_calls":0,"customer_files_used":0},ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-queue",action="store_true",help="Assert that each review completed through the database-backed RabbitMQ worker")
    main(parser.parse_args().require_queue)
